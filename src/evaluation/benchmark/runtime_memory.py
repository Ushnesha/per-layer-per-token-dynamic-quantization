"""Worker-side allocation lifetimes and scheduler-assigned KV occupancy."""

from types import MethodType

TRACE_LIMIT = 50000


def transient_allocations(trace):
    """Peak concurrent allocations born and freed within a measured model step."""
    live, lifetimes = {}, []
    for index, event in enumerate(trace):
        address = event.get('addr')
        if event['action'] == 'alloc':
            if address in live:
                raise ValueError('Allocation address reused before completed free.')
            live[address] = (index, int(event['size']))
        elif event['action'] == 'free_completed' and address in live:
            start, size = live.pop(address)
            lifetimes.append((start, index, size))
    points = []
    for start, end, size in lifetimes:
        points.extend(((start, size), (end, -size)))
    current = peak = 0
    for _, delta in sorted(points):
        current += delta
        peak = max(peak, current)
    return {'peak_concurrent_bytes': peak, 'allocation_count': len(lifetimes),
            'cumulative_allocated_bytes': sum(row[2] for row in lifetimes),
            'allocations_remaining_at_step_end': len(live)}


def update_blocks(state, output):
    for request in set(output.finished_req_ids) | set(getattr(output, 'preempted_req_ids', None) or ()):
        state['requests'].pop(request, None)
    for request in output.scheduled_new_reqs:
        state['requests'][request.req_id] = [list(group) for group in request.block_ids]
    cached = output.scheduled_cached_reqs
    for request, groups in zip(cached.req_ids, cached.new_block_ids):
        if groups is None:
            continue
        if request in cached.resumed_req_ids:
            state['requests'][request] = [list(group) for group in groups]
        else:
            existing = state['requests'].setdefault(request, [[] for _ in groups])
            if len(existing) != len(groups):
                raise ValueError('KV group count changed within a request.')
            for old, new in zip(existing, groups):
                old.extend(new)


def install_memory_tracking(worker):
    import torch
    runner = worker.model_runner
    if hasattr(worker, '_aditya_memory_tracking'):
        raise RuntimeError('Memory tracking already installed.')
    if not callable(getattr(torch.cuda.memory, '_record_memory_history', None)):
        raise RuntimeError('Installed Torch does not expose allocator history.')
    config = runner.kv_cache_config
    if len(config.kv_cache_groups) != 1:
        raise RuntimeError('Live byte accounting requires the single-group cache adapter; hybrid cache layouts need an adapter.')
    # The pool was counted by real tensor storage accounting, not nominal bits.
    from .component_memory import collect_worker_components
    persistent = collect_worker_components(worker)
    pool = persistent['components']['kv_cache_allocated_pool']['bytes']
    if config.num_blocks <= 0 or pool % config.num_blocks:
        raise ValueError('KV pool is not an integral number of physical blocks.')
    state = {'original': runner.execute_model, 'requests': {}, 'steps': [],
             'bytes_per_block': pool // config.num_blocks, 'pool_bytes': pool,
             'cache_dtype': str(runner.kv_cache_dtype),
             'num_blocks': config.num_blocks, 'block_size': config.kv_cache_groups[0].kv_cache_spec.block_size}
    worker._aditya_memory_tracking = state

    def execute(self, scheduler_output, *args, **kwargs):
        update_blocks(state, scheduler_output)
        scheduled = scheduler_output.num_scheduled_tokens
        if not scheduled:
            return state['original'](scheduler_output, *args, **kwargs)
        # Prefix caching is disabled, so these are pages assigned to active requests.
        blocks = set()
        for request in scheduled:
            groups = state['requests'].get(request)
            if groups is None or len(groups) != 1:
                raise ValueError('Scheduled request has no complete cache block inventory.')
            blocks.update(groups[0])
        if any(type(block) is not int or block < 0 or block >= state['num_blocks'] for block in blocks):
            raise ValueError('Scheduler returned an invalid cache block ID.')
        torch.cuda.synchronize(worker.device)
        before = torch.cuda.memory_allocated(worker.device)
        torch.cuda.reset_peak_memory_stats(worker.device)
        torch.cuda.memory._record_memory_history(enabled='all', context=None,
                                                max_entries=TRACE_LIMIT, clear_history=True)
        try:
            result = state['original'](scheduler_output, *args, **kwargs)
            torch.cuda.synchronize(worker.device)
            peak = torch.cuda.max_memory_allocated(worker.device)
            reserved_peak = torch.cuda.max_memory_reserved(worker.device)
            after = torch.cuda.memory_allocated(worker.device)
            traces = torch.cuda.memory._snapshot()['device_traces']
            device_index = worker.device.index if worker.device.index is not None else torch.cuda.current_device()
            trace = traces[device_index]
            if len(trace) >= TRACE_LIMIT:
                raise RuntimeError('Allocator trace truncated; refusing incomplete transient accounting.')
            transient = transient_allocations(trace)
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        state['steps'].append({'processed_tokens': sum(scheduled.values()),
            'phase': 'prefill' if max(scheduled.values()) > 1 else 'decode',
            'active_requests': len(scheduled), 'occupied_blocks': len(blocks),
            'occupied_kv_bytes': len(blocks) * state['bytes_per_block'],
            'allocator_before_bytes': before, 'allocator_after_bytes': after,
            'allocator_peak_bytes': peak, 'transient_allocations': transient,
            'allocator_reserved_peak_bytes': reserved_peak,
            'active_working_set_peak_bytes': peak - state['pool_bytes'] + len(blocks) * state['bytes_per_block']})
        return result

    runner.execute_model = MethodType(execute, runner)
    return {'status': 'ok'}


def remove_memory_tracking(worker):
    import torch
    state = worker._aditya_memory_tracking
    worker.model_runner.execute_model = state['original']
    torch.cuda.memory._record_memory_history(enabled=None)
    del worker._aditya_memory_tracking
    if not state['steps']:
        return {'status': 'failed', 'reason': 'No measured model steps were recorded.'}
    return {'status': 'ok', 'steps': state['steps'],
            'accounting_version': 1, 'cache_dtype': state['cache_dtype'],
            'pool_bytes': state['pool_bytes'], 'physical_bytes_per_block': state['bytes_per_block'],
            'tokens_per_block': state['block_size'],
            'peak_occupied_blocks': max(row['occupied_blocks'] for row in state['steps']),
            'peak_occupied_kv_bytes': max(row['occupied_kv_bytes'] for row in state['steps']),
            'peak_transient_bytes': max(row['transient_allocations']['peak_concurrent_bytes'] for row in state['steps']),
            'peak_allocator_bytes': max(row['allocator_peak_bytes'] for row in state['steps']),
            'peak_allocator_reserved_bytes': max(row['allocator_reserved_peak_bytes'] for row in state['steps']),
            'peak_active_working_set_bytes': max(row['active_working_set_peak_bytes'] for row in state['steps']),
            'scope': 'scheduler-assigned KV pages and PyTorch allocator lifetimes inside execute_model; physical KV pages include block padding',
            'transient_scope': 'temporary PyTorch allocations, including activations/intermediates/workspaces; CUDA-graph retained buffers are persistent, not zero-cost workspaces; non-PyTorch allocations are excluded'}
