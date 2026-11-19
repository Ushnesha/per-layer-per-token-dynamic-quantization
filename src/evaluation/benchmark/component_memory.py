"""Checkpoint decomposition plus worker-side persistent GPU storage snapshots."""

from .worker_extension import WORKER_EXTENSION

from pathlib import Path

from .checkpoint_storage import inspect_checkpoint, tensor_category

GIB = 1024 ** 3


def checkpoint_components(model_path: Path) -> dict:
    inventory = inspect_checkpoint(model_path)
    if inventory is None:
        return {'status': 'unsupported', 'reason': 'No safetensors checkpoint; pickle-based checkpoints are not loaded for accounting.'}
    categories = {}
    for tensor in inventory['tensors']:
        categories[tensor['category']] = categories.get(tensor['category'], 0) + tensor['bytes']
    return {'status': 'ok', **inventory,
            'components': {name: {'bytes': size, 'gib': size / GIB} for name, size in categories.items()},
            'classification': 'explicit tensor-name suffix rules; unknown names retained as other_tensors',
            'scope': 'checkpoint storage; not runtime GPU allocations'}


def summarize_storages(records: list[dict], allocated_bytes: int, reserved_bytes: int) -> dict:
    """Count full backing storage once, including tied weights and cache views."""
    unique = {}
    for record in records:
        if record['bytes'] < 0:
            raise ValueError('Storage size cannot be negative.')
        key = (record['device'], record['storage_id'])
        entry = unique.setdefault(key, {'bytes': record['bytes'], 'names': [], 'categories': set()})
        if entry['bytes'] != record['bytes']:
            raise ValueError('Aliases disagree on the size of their shared storage.')
        entry['names'].append(record['name'])
        entry['categories'].add(record['category'])
    categories = {}
    details = []
    for entry in unique.values():
        names = sorted(entry['categories'])
        category = names[0] if len(names) == 1 else 'shared_across_components'
        categories[category] = categories.get(category, 0) + entry['bytes']
        details.append({'bytes': entry['bytes'], 'names': entry['names'], 'categories': names})
    visible = sum(entry['bytes'] for entry in unique.values())
    if allocated_bytes < visible or reserved_bytes < allocated_bytes:
        raise ValueError('Worker allocator snapshot is inconsistent with visible storage.')
    if allocated_bytes > visible:
        categories['unattributed_live_allocator_allocations'] = allocated_bytes - visible
    return {
        'components': {name: {'bytes': size, 'gib': size / GIB} for name, size in categories.items()},
        'visible_storage_bytes': visible, 'worker_cuda_allocated_bytes': allocated_bytes,
        'worker_cuda_reserved_bytes': reserved_bytes,
        'unattributed_allocated_bytes': allocated_bytes - visible,
        'allocator_reserved_but_unallocated_bytes': reserved_bytes - allocated_bytes,
        'unattributed_scope': 'measured allocator remainder, including opaque retained graph/runtime buffers and allocator rounding; not assumed to be workspace',
        'unique_storages': details,
    }


def collect_worker_components(worker) -> dict:
    """Callable executed by vLLM.collective_rpc on the actual GPU worker."""
    import torch
    runner = getattr(worker, 'model_runner', None)
    model = getattr(runner, 'model', None)
    if model is None:
        return {'status': 'unsupported', 'reason': 'Worker model_runner.model is not exposed.'}
    device = worker.device
    torch.cuda.synchronize(device)
    records = []
    def add(name, tensor, category):
        if not torch.is_tensor(tensor) or tensor.device.type != 'cuda':
            return
        storage = tensor.untyped_storage()
        records.append({'name': name, 'category': category, 'device': str(tensor.device),
                        'storage_id': storage.data_ptr(), 'bytes': storage.nbytes()})
    for name, tensor in model.named_parameters():
        category = tensor_category(name)
        add(name, tensor, category if category != 'other_tensors' else 'other_model_parameters')
    for name, tensor in model.named_buffers():
        category = tensor_category(name)
        add(name, tensor, category if category != 'other_tensors' else 'persistent_model_buffers')
    seen = set()
    def visit(value, name, category, depth=0):
        if torch.is_tensor(value):
            add(name, value, category)
        elif id(value) in seen or depth > 6:
            return
        elif isinstance(value, (list, tuple)):
            seen.add(id(value))
            for index, item in enumerate(value):
                visit(item, f'{name}/{index}', category, depth + 1)
        elif isinstance(value, dict):
            seen.add(id(value))
            for key, item in value.items():
                visit(item, f'{name}/{key}', category, depth + 1)
        elif hasattr(value, '__dict__') and not isinstance(value, torch.nn.Module):
            seen.add(id(value))
            visit(vars(value), name, category, depth + 1)
    caches = getattr(runner, 'kv_caches', None)
    if caches is not None:
        visit(caches, 'model_runner.kv_caches', 'kv_cache_allocated_pool')
    # Inspect only the known GPU metadata/input holders, not the whole runner.
    visit(getattr(runner, 'block_tables', None), 'model_runner.block_tables', 'kv_cache_metadata')
    visit(getattr(runner, 'input_buffers', None), 'model_runner.input_buffers', 'execution_input_buffers')
    result = summarize_storages(records, torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device))
    cache_config = getattr(runner, 'kv_cache_config', None)
    return {
        'status': 'ok', **result,
        'kv_pool_visibility': 'exposed tensor storages' if caches is not None else 'unsupported',
        'configured_cache_blocks': getattr(cache_config, 'num_blocks', None),
        'live_kv_occupancy': {'status': 'unsupported', 'reason': 'An idle worker snapshot does not expose active scheduler block occupancy.'},
        'temporary_workspaces': {'status': 'unsupported', 'reason': 'Transient kernel workspaces cannot be attributed from persistent tensor snapshots.'},
    }


def snapshot_components(engine, tracking=None) -> dict:
    """Read persistent storage from an existing engine without loading another."""
    if not callable(getattr(engine, 'collective_rpc', None)):
        return {'status': 'unsupported', 'reason': 'This vLLM build does not expose collective_rpc.'}
    try:
        workers = engine.collective_rpc('aditya_collect_components', timeout=60)
    except Exception as error:
        return {'status': 'unsupported', 'reason': f'Worker memory introspection unavailable: {error}'}
    if len(workers) != 1:
        return {'status': 'unsupported', 'reason': 'Component accounting currently requires exactly one GPU worker.'}
    result = {**workers[0], 'snapshot_phase': 'after completed generation; persistent allocations only'}
    if tracking is not None and tracking.get('status') == 'ok':
        result['live_kv_occupancy'] = {'status': 'ok',
            'peak_bytes': tracking['peak_occupied_kv_bytes'],
            'peak_blocks': tracking['peak_occupied_blocks'],
            'bytes_per_block': tracking['physical_bytes_per_block'],
            'tokens_per_block': tracking['tokens_per_block'],
            'scope': 'actual scheduler-assigned physical pages, including block padding; excludes unused pool capacity'}
        result['temporary_workspaces'] = {'status': 'ok',
            'peak_bytes': tracking['peak_transient_bytes'], 'scope': tracking['transient_scope']}
        result['measured_execution'] = tracking
    return result


def runtime_components(model_path: str) -> dict:
    from .peak_memory import evaluate_peak_memory
    measured = evaluate_peak_memory(model_path, include_components=True)
    return measured.get('component_snapshot', {'status': 'unsupported', 'reason': measured.get('reason', 'No component measurements.')})


def evaluate_component_memory(model_path: str) -> dict:
    checkpoint = checkpoint_components(Path(model_path))
    runtime = runtime_components(model_path)
    return {
        'status': 'ok' if checkpoint['status'] == runtime['status'] == 'ok' else 'unsupported',
        'coverage': 'persistent tensor storage, occupied KV pages and transient PyTorch allocations; non-PyTorch workspaces excluded',
        'checkpoint': checkpoint, 'runtime': runtime,
        'accounting_rules': [
            'Checkpoint and runtime values describe different representations and must not be added.',
            'GPU storage is counted once; aliases across categories are reported as shared.',
            'CUDA reserved memory includes allocated memory; the values are not additive.',
            'Unattributed allocator bytes are not labelled as workspace, cache metadata, or driver memory.',
            'KV pool allocation is capacity, not live request occupancy.',
        ],
    }
