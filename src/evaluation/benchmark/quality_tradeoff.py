"""Quality changes versus a compatible FP16 run and measured weight storage.

Uses saved runs rather than quantizing checkpoints or changing their precision.
Memory axes retain their own measured workloads; no precision sweep is invented.
"""

import hashlib
import json
import math
from pathlib import Path


def checkpoint_tensor_bytes(model_path: Path):
    """Count safetensors data offsets, including packed weights and scale tensors."""
    from .checkpoint_storage import inspect_checkpoint
    result = inspect_checkpoint(model_path)
    return result['tensor_bytes'] if result is not None else None


def describe_checkpoint(model_path: Path) -> dict:
    config = json.loads((model_path / 'config.json').read_text())
    # Quantization and serialization metadata may differ for the same base model.
    ignored = {'quantization_config', 'torch_dtype', 'dtype', '_name_or_path', 'transformers_version'}
    base = {key: value for key, value in config.items() if key not in ignored}
    identity = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    digest = hashlib.sha256()
    token_files = [model_path / name for name in
                   ('tokenizer.json', 'tokenizer.model', 'tokenizer_config.json', 'special_tokens_map.json')
                   if (model_path / name).is_file()]
    for file in token_files:
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    quantization = config.get('quantization_config') or {}
    try:
        tensor_bytes = checkpoint_tensor_bytes(model_path)
        storage_error = None
    except (OSError, ValueError, KeyError, TypeError) as error:
        tensor_bytes, storage_error = None, str(error)
    return {
        'base_config_sha256': identity,
        'tokenizer_sha256': digest.hexdigest() if token_files else None,
        'weight_method': quantization.get('quant_method', 'unquantized' if not quantization else 'unknown'),
        'quantization_config': quantization,
        'weight_tensor_bytes': tensor_bytes,
        'weight_tensor_gib': tensor_bytes / 1024 ** 3 if tensor_bytes is not None else None,
        'storage_error': storage_error,
        'storage_scope': 'checkpoint tensor payload, including quantization tensors; excludes file headers and GPU/KV allocations',
    }


def quality_scores(run: dict) -> dict:
    """Return score, direction, and measurement protocol for comparable outputs."""
    scores = {}
    for metric_name, metric in run.get('metrics', {}).items():
        if metric.get('status') != 'ok':
            continue
        if metric_name == 'perplexity':
            fields = ('dataset', 'dataset_config', 'dataset_fingerprint', 'split', 'context_length',
                      'num_windows', 'scored_tokens', 'bos_token_id', 'protocol', 'scoring_path')
            scores['perplexity'] = (metric['value'], 'lower', {key: metric.get(key) for key in fields})
        elif metric_name == 'accuracy':
            fields = ('task', 'dataset', 'split', 'num_fewshot', 'apply_chat_template', 'seed',
                      'num_examples', 'unit', 'task_version', 'scoring')
            for key in ('accuracy', 'normalized_accuracy'):
                scores[key] = (metric[key], 'higher', {field: metric.get(field) for field in fields})
        elif metric_name == 'retrieval':
            fields = ('benchmark', 'seed', 'examples_per_cell', 'output_tokens', 'apply_chat_template',
                      'scoring', 'position_definition', 'engine_settings', 'prompting_version',
                      'chat_template_sha256', 'stop_token_ids', 'stop_strings')
            protocol = {key: metric.get(key) for key in fields}
            for cell in metric['by_context_and_position']:
                key = f"retrieval/{cell['context_length']}/{cell['target_position']}"
                scores[key] = (cell['accuracy'], 'higher', {**protocol, 'total': cell['total']})
        elif metric_name == 'long_context':
            fields = ('benchmark', 'archive_sha256', 'split', 'apply_chat_template', 'protocol', 'engine_settings',
                      'prompting_version', 'chat_template_sha256', 'stop_token_ids')
            protocol = {key: metric.get(key) for key in fields}
            for task, result in metric['tasks'].items():
                for cell in result['by_input_budget']:
                    key = f"long_context/{task}/{cell['input_budget']}"
                    scores[key] = (cell['score'], 'higher', {**protocol, 'metric': result['metric'],
                                  'output_tokens': result['output_tokens'], 'num_examples': cell['num_examples']})
    for value, _, _ in scores.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Quality comparison requires finite numeric scores.')
    return scores


def compatibility_reason(current: dict, other: dict):
    if current.get('evaluation_profile', 'full') != other.get('evaluation_profile', 'full'):
        return 'Different evaluation_profile'
    for key in ('base_config_sha256', 'tokenizer_sha256'):
        value = current.get('checkpoint', {}).get(key)
        if not value or value != other.get('checkpoint', {}).get(key):
            return f'Missing or different {key}'
    for key in ('backend', 'engine_settings', 'versions', 'runtime_settings'):
        if current.get(key) != other.get(key):
            return f'Different {key}'
    return None


def memory_budgets(run):
    """Keep memory axes and the workload supporting each axis explicit."""
    checkpoint = run.get('checkpoint', {})
    budgets = {}
    if checkpoint.get('weight_tensor_bytes') is not None:
        budgets['weight_tensor'] = {'bytes': checkpoint['weight_tensor_bytes'],
            'protocol': 'all checkpoint tensor bytes including metadata/padding'}
    from .compression_ratio import runtime_measurements
    measured = runtime_measurements(run)
    if measured is not None:
        trace = measured['trace']
        budgets['active_runtime_tensors'] = {'bytes': trace['peak_active_working_set_bytes'],
            'protocol': measured['protocol']}
        if measured['metadata'] is not None and trace['cache_dtype'] in ('torch.float16', 'torch.bfloat16'):
            budgets['occupied_kv_with_metadata'] = {'bytes': trace['peak_occupied_kv_bytes'] + measured['metadata'],
                'protocol': measured['protocol']}
    return budgets


def is_fp16_reference(run):
    metrics = run.get('metrics', {})
    trace = metrics.get('component_memory', {}).get('runtime', {}).get('measured_execution', {})
    instrumentation = metrics.get('kernel_time', {}).get('cuda_events', {}).get('adaptive_instrumentation', {})
    return (run.get('checkpoint', {}).get('weight_method') == 'unquantized'
            and run.get('engine_settings', {}).get('dtype') == 'float16'
            and run.get('engine_settings', {}).get('kv_cache_dtype', 'auto') in ('auto', 'float16')
            and trace.get('cache_dtype', 'torch.float16') == 'torch.float16'
            and instrumentation.get('status') != 'ok')


def compare_quality(current: dict, previous: list[tuple[str, dict]], reference_source=None) -> dict:
    """Use the newest compatible unquantized/FP16 result; retain precision points."""
    candidates = [*previous, ('current_run', current)]
    compatible = []
    excluded = []
    for source, run in candidates:
        reason = compatibility_reason(current, run)
        if reason:
            excluded.append({'source': source, 'reason': reason})
        else:
            compatible.append((source, run))
    references = [(source, run) for source, run in compatible
                  if is_fp16_reference(run)
                  and (reference_source is None or source == reference_source)]
    if not references:
        return {'status': 'blocked', 'reason': 'Run the matching FP16 checkpoint first; no compatible reference is saved.',
                'excluded_runs': excluded}
    source, reference = max(references, key=lambda item: item[1].get('timestamp', ''))
    reference_scores = quality_scores(reference)
    points = []
    mismatches = []
    for run_source, run in compatible:
        for name, (score, direction, protocol) in quality_scores(run).items():
            if name not in reference_scores or protocol != reference_scores[name][2]:
                mismatches.append({'source': run_source, 'metric': name, 'reason': 'Missing reference score or different measurement protocol'})
                continue
            baseline = reference_scores[name][0]
            memory = run['checkpoint'].get('weight_tensor_gib')
            points.append({
                'source': run_source, 'model': run['model'], 'weight_method': run['checkpoint']['weight_method'],
                'quantization_config': run['checkpoint'].get('quantization_config', {}),
                'metric': name, 'score': score, 'reference_score': baseline,
                'quality_loss': score - baseline if direction == 'lower' else baseline - score,
                'quality_loss_unit': 'PPL' if direction == 'lower' else 'score points',
                'weight_tensor_gib': memory, 'memory_available': memory is not None,
                'measured_memory_budgets': memory_budgets(run),
            })
    pairs = []
    budget_pairs = []
    for index, first in enumerate(points):
        for second in points[index + 1:]:
            if first['metric'] != second['metric'] or first['source'] == second['source']:
                continue
            a, b = first['weight_tensor_gib'], second['weight_tensor_gib']
            if a is not None and b is not None and max(a, b) > 0 and abs(a - b) / max(a, b) <= 0.05:
                pairs.append({'metric': first['metric'], 'sources': [first['source'], second['source']],
                              'relative_memory_difference': abs(a - b) / max(a, b)})
            for axis, left in first['measured_memory_budgets'].items():
                right = second['measured_memory_budgets'].get(axis)
                if right is None or left['protocol'] != right['protocol']:
                    continue
                maximum = max(left['bytes'], right['bytes'])
                difference = abs(left['bytes'] - right['bytes']) / maximum if maximum else 0
                if difference <= 0.05:
                    budget_pairs.append({'metric': first['metric'], 'memory_axis': axis,
                        'sources': [first['source'], second['source']], 'relative_memory_difference': difference,
                        'quality_losses': [first['quality_loss'], second['quality_loss']]})
    return {
        'status': 'ok' if points else 'blocked',
        'reason': None if points else 'No scores share the reference measurement protocol.',
        'reference_source': source, 'reference_model': reference['model'],
        'comparison_scope': 'reference_only' if all(is_fp16_reference(run)
            for _, run in compatible) else 'measured_precision_points',
        'points': points, 'matched_weight_storage_pairs': pairs, 'memory_tolerance_percent': 5,
        'matched_memory_budget_pairs': budget_pairs,
        'excluded_runs': excluded, 'excluded_measurements': mismatches,
        'interpretation': 'Positive loss means worse quality; negative loss means an improvement. Plot-ready data points are embedded in this JSON.',
        'limitations': [
            'Weight storage and runtime/KV axes are distinct; runtime/KV measurements use the recorded fixed diagnostic workload, not per-task peaks.',
            'Architecture/tokenizer matching does not prove the same training lineage; compare checkpoints derived from the same base model.',
            'HellaSwag comparison assumes the same dataset content under the recorded protocol and software versions.',
            'A reference-only run is not a precision-budget experiment; quantized/adaptive comparisons require actual supported backend checkpoints.',
        ],
    }


def saved_runs(results_dir: Path, reference_path=None):
    previous = []
    unreadable = []
    files = sorted(results_dir.glob('evaluation_*.json'))
    if reference_path is not None:
        files = [file for file in files if file.resolve() != reference_path] + [reference_path]
    for file in files:
        try:
            run = json.loads(file.read_text())
            if isinstance(run, dict) and run.get('run_status') in ('running', 'failed'):
                continue
            if not isinstance(run, dict):
                raise ValueError('Saved evaluation must be a JSON object.')
            previous.append((str(file), run))
        except (OSError, ValueError) as error:
            unreadable.append({'source': str(file), 'reason': str(error)})
    return previous, unreadable


def validate_reference(current: dict, path: Path):
    run = json.loads(path.read_text())
    if not isinstance(run, dict) or run.get('run_status') not in ('ok', 'partial'):
        raise ValueError('Reference must be a finished successful or partial evaluation report.')
    reason = compatibility_reason(current, run)
    if reason:
        raise ValueError(f'Incompatible FP16 reference: {reason}')
    if not is_fp16_reference(run):
        raise ValueError('Reference must measure an unquantized FP16 checkpoint.')
    if not quality_scores(run):
        raise ValueError('Reference contains no successful quality measurements.')


def evaluate_quality_tradeoff(current: dict, results_dir: Path, reference_path=None) -> dict:
    previous, unreadable = saved_runs(results_dir, reference_path)
    result = compare_quality(current, previous, str(reference_path) if reference_path else None)
    result['unreadable_runs'] = unreadable
    return result
