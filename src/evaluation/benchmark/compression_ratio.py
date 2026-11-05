"""Effective storage ratios against compatible saved FP16 measurements.

No nominal bit-width ratios and no inference from vLLM's preallocated KV pool.
"""

import math
from pathlib import Path

from .quality_tradeoff import compatibility_reason, is_fp16_reference, saved_runs


def storage_ratio(reference_bytes: int, compressed_bytes: int) -> dict:
    """Include every measured representation byte, not just its weight payload."""
    if any(type(value) is not int or value <= 0 for value in (reference_bytes, compressed_bytes)):
        raise ValueError('Compression requires positive integer byte counts for both representations.')
    ratio = reference_bytes / compressed_bytes
    if not math.isfinite(ratio):
        raise ValueError('Compression ratio must be finite.')
    return {
        'status': 'ok', 'value': ratio, 'unit': 'x',
        'reference_bytes': reference_bytes, 'compressed_bytes': compressed_bytes,
        'reference_gib': reference_bytes / 1024 ** 3,
        'compressed_gib': compressed_bytes / 1024 ** 3,
        'storage_reduction_percent': 100 * (1 - compressed_bytes / reference_bytes),
    }


def checkpoint_sizes(run: dict) -> dict:
    metric = run.get('metrics', {}).get('component_memory', {})
    checkpoint = metric.get('checkpoint', {})
    if checkpoint.get('status') == 'ok':
        return {'tensor_bytes': checkpoint.get('tensor_bytes'), 'file_bytes': checkpoint.get('file_bytes')}
    # Older runs from this evaluator can still support the tensor-only ratio.
    return {'tensor_bytes': run.get('checkpoint', {}).get('weight_tensor_bytes'), 'file_bytes': None}


def runtime_measurements(run):
    metrics = run.get('metrics', {})
    runtime = metrics.get('component_memory', {}).get('runtime', {})
    trace = runtime.get('measured_execution', {})
    peak = metrics.get('peak_memory', {})
    if runtime.get('status') != 'ok' or trace.get('status') != 'ok' or peak.get('status') != 'ok':
        return None
    metadata = runtime.get('components', {}).get('kv_cache_metadata', {}).get('bytes')
    return {'trace': trace, 'peak': peak, 'metadata': metadata,
            'protocol': {'workload': trace['workload'], 'accounting_version': trace['accounting_version'],
                         'engine_settings': peak.get('engine_settings'),
                         'gpu_name': peak.get('gpu_name'), 'gpu_total_gib': peak.get('gpu_total_gib'),
                         'steps': [(step['phase'], step['processed_tokens'], step['active_requests'])
                                   for step in trace['steps']]}}


def compare_compression(current: dict, previous: list[tuple[str, dict]], reference_source=None) -> dict:
    candidates = [*previous, ('current_run', current)]
    references = []
    excluded = []
    for source, run in candidates:
        reason = compatibility_reason(current, run)
        if reason:
            excluded.append({'source': source, 'reason': reason})
        elif (is_fp16_reference(run)
              and (reference_source is None or source == reference_source)):
            references.append((source, run))
    if not references:
        return {'status': 'blocked', 'reason': 'No compatible FP16 reference; evaluate the matching FP16 checkpoint first.',
                'excluded_runs': excluded}
    source, reference = max(references, key=lambda item: item[1].get('timestamp', ''))
    ref_sizes, current_sizes = checkpoint_sizes(reference), checkpoint_sizes(current)
    components = {}
    for name, key, scope in (
        ('weight_tensor_storage', 'tensor_bytes',
         'all checkpoint tensors, including quantization metadata and physical packing/padding; excludes shard headers'),
        ('weight_shard_files', 'file_bytes',
         'safetensors shard files, including headers and file overhead; excludes tokenizer/config and other files'),
    ):
        a, b = ref_sizes[key], current_sizes[key]
        if a is None or b is None:
            components[name] = {'status': 'blocked', 'reason': 'Measured bytes missing in candidate or reference.'}
        else:
            components[name] = {**storage_ratio(a, b), 'scope': scope}

    a, b = runtime_measurements(reference), runtime_measurements(current)
    comparable = a is not None and b is not None and a['protocol'] == b['protocol']
    for name in ('kv_cache', 'total_runtime_footprint', 'active_runtime_tensor_footprint'):
        components[name] = {'status': 'blocked',
            'reason': 'Reference and candidate require successful matching memory instrumentation, workload, hardware, and engine settings.'}
    if comparable:
        # Do not divide the automatically sized unused KV pools. FP16/BF16 has
        # no extra quantization scales; other layouts need a byte-accounting adapter.
        if (a['metadata'] is not None and b['metadata'] is not None
                and all(row['trace']['cache_dtype'] in ('torch.float16', 'torch.bfloat16') for row in (a, b))):
            components['kv_cache'] = {**storage_ratio(
                a['trace']['peak_occupied_kv_bytes'] + a['metadata'],
                b['trace']['peak_occupied_kv_bytes'] + b['metadata']),
                'scope': 'occupied physical KV pages including block padding, plus allocated GPU cache block/slot metadata; excludes unused data pages',
                'reference_occupied_page_bytes': a['trace']['peak_occupied_kv_bytes'],
                'candidate_occupied_page_bytes': b['trace']['peak_occupied_kv_bytes'],
                'reference_metadata_bytes': a['metadata'], 'candidate_metadata_bytes': b['metadata']}
        else:
            components['kv_cache']['reason'] = 'Cache metadata is missing or the quantized cache layout needs its backend byte-accounting adapter.'
        components['total_runtime_footprint'] = {**storage_ratio(
            round(a['peak']['peak_increase_over_baseline_gib'] * 1024 ** 3),
            round(b['peak']['peak_increase_over_baseline_gib'] * 1024 ** 3)),
            'scope': 'sampled total device peak growth across startup and inference; includes unused KV pool and driver/runtime overhead',
            'interpretation': 'Observed resident-footprint ratio; pool sizing can mask tensor compression. Sampling can miss brief peaks.'}
        components['active_runtime_tensor_footprint'] = {**storage_ratio(
            a['trace']['peak_active_working_set_bytes'], b['trace']['peak_active_working_set_bytes']),
            'scope': 'peak worker PyTorch allocated bytes, replacing unused KV data-pool capacity with occupied physical pages; excludes non-PyTorch/driver allocations'}
    return {
        'status': 'ok' if any(value['status'] == 'ok' for value in components.values()) else 'blocked',
        'coverage': 'measured checkpoint, occupied KV with metadata, resident runtime, and active runtime tensor ratios; see each representation status/scope',
        'comparison_scope': 'reference_measurement' if source == 'current_run' else 'measured_reference_comparison',
        'reference_source': source, 'reference_model': reference['model'],
        'model': current['model'], 'components': components, 'excluded_runs': excluded,
        'formula': 'reference_bytes / compressed_bytes',
        'interpretation': '1x means equal storage; >1x means compression; <1x means increased storage.',
        'limitations': [
            'Checkpoint storage is distinct from vLLM repacked GPU weights.',
            'Model/tokenizer matching does not establish common training lineage; use checkpoints derived from the same base model.',
            'No nominal 16/bits estimate substitutes for measured storage.',
        ],
    }


def evaluate_compression_ratio(current: dict, results_dir: Path, reference_path=None) -> dict:
    previous, unreadable = saved_runs(results_dir, reference_path)
    result = compare_compression(current, previous, str(reference_path) if reference_path else None)
    result['unreadable_runs'] = unreadable
    return result
