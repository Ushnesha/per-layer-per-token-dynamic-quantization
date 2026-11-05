"""Human-readable Markdown summaries of the evaluator's existing JSON results."""

import html
import math
from pathlib import Path
from urllib.parse import quote

METRICS = (
    ('Model effectiveness', 'perplexity', 'Perplexity'),
    ('Model effectiveness', 'accuracy', 'Downstream task accuracy'),
    ('Model effectiveness', 'retrieval', 'Long-context retrieval accuracy'),
    ('Model effectiveness', 'long_context', 'Long-context task performance'),
    ('Model effectiveness', 'quality_tradeoff', 'Quality degradation across precision budgets'),
    ('Hardware efficiency', 'peak_memory', 'Peak GPU memory'),
    ('Hardware efficiency', 'component_memory', 'Component-wise memory footprint'),
    ('Hardware efficiency', 'compression_ratio', 'Effective compression ratio'),
    ('Hardware efficiency', 'kernel_time', 'GPU kernel execution time'),
    ('Hardware efficiency', 'adaptive_overhead', 'Adaptive quantization overhead'),
    ('Serving efficiency', 'ttft', 'Time to first token'),
    ('Serving efficiency', 'tpot', 'Time per output token'),
    ('Serving efficiency', 'e2e_latency', 'End-to-end request latency'),
    ('Serving efficiency', 'throughput', 'Serving throughput'),
    ('Serving efficiency', 'goodput', 'Request goodput'),
    ('Serving efficiency', 'max_concurrency', 'Maximum sustainable concurrency'),
    ('Serving efficiency', 'max_context', 'Maximum memory-feasible context length'),
)


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'Yes' if value else 'No'
    if isinstance(value, float):
        if not math.isfinite(value):
            return '—'
        value = f'{value:.4g}' if value != 0 and abs(value) < 0.005 else f'{value:,.2f}'.rstrip('0').rstrip('.')
    elif isinstance(value, int):
        value = f'{value:,}'
    return html.escape(str(value), quote=False).replace('\\', '\\\\').replace('|', r'\|').replace('\r\n', '\n').replace('\n', '<br>')


def table(headers, rows):
    rows = list(rows)
    if not rows:
        return '_Pending._\n'
    lines = ['| ' + ' | '.join(map(cell, headers)) + ' |',
             '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in rows:
        if len(row) != len(headers):
            raise ValueError('Report row does not match its table headings.')
        lines.append('| ' + ' | '.join(map(cell, row)) + ' |')
    return '\n'.join(lines) + '\n'



def value(metric, key):
    return metric.get(key) if metric.get('status') in ('ok', 'partial') else None


def scaled(amount, divisor):
    return amount / divisor if isinstance(amount, (int, float)) else None


def method_label(results):
    checkpoint = results.get('checkpoint', {})
    method = checkpoint.get('weight_method', 'unknown')
    dtype = results.get('engine_settings', {}).get('dtype')
    if method == 'unquantized':
        return {'float16': 'FP16', 'bfloat16': 'BF16', 'float32': 'FP32'}.get(dtype, dtype or 'Unquantized')
    return str(method).upper()


def precision_bits(results):
    checkpoint = results.get('checkpoint', {})
    config = checkpoint.get('quantization_config', {})
    for name in ('bits', 'w_bit', 'weight_bits'):
        if name in config:
            return config[name]
    if checkpoint.get('weight_method') == 'unquantized':
        return {'float16': 16, 'bfloat16': 16, 'float32': 32}.get(results.get('engine_settings', {}).get('dtype'))
    return None


def serving_row(metric, input_tokens):
    # A fixed representative workload, never the fastest trial or a blend of loads.
    if metric.get('status') not in ('ok', 'partial'):
        return {}
    return next((row for row in metric.get('by_workload', [])
                 if row.get('input_tokens') == input_tokens and row.get('output_tokens') == 128
                 and row.get('concurrency') == 1 and row.get('request_rate') is None
                 and row.get('status') == 'ok'), {})


def latency_pair(row):
    return f"{cell(row.get('p50'))} / {cell(row.get('p95'))}"


def render_markdown(results: dict, json_name: str) -> str:
    metrics = results.get('metrics', {})
    method = method_label(results)
    checkpoint = results.get('checkpoint', {})
    runtime = metrics.get('component_memory', {}).get('runtime', {})
    components = runtime.get('components', {})
    quality = metrics.get('quality_tradeoff', {})
    baseline_only = quality.get('comparison_scope') == 'reference_only'
    parts = [f"# {cell(Path(results.get('model', 'Model')).name)}\n\n",
             f"{cell(results.get('run_status', 'running'))} · [JSON]({quote(json_name)})\n\n",
             table(['Method', 'Bits', 'WikiText-2 PPL ↓', 'HellaSwag acc_norm (%) ↑',
                    'Weights (GiB)', 'TPOT p50 (ms) ↓'], [[
                 method, precision_bits(results), value(metrics.get('perplexity', {}), 'value'),
                 value(metrics.get('accuracy', {}), 'normalized_accuracy'),
                 checkpoint.get('weight_tensor_gib'), serving_row(metrics.get('tpot', {}), 512).get('p50'),
             ]]),
             '\nTPOT: 512 input / 128 output tokens, concurrency 1.\n']

    # Keep context/position differences, without per-example rows or explanations.
    retrieval = metrics.get('retrieval', {})
    long_context = metrics.get('long_context', {})
    contexts = sorted({row['context_length'] for row in retrieval.get('by_context_and_position', [])}
                      | {row['input_budget'] for task in long_context.get('tasks', {}).values()
                         for row in task.get('by_input_budget', [])})
    quality_rows = []
    positions = sorted({row['target_position'] for row in retrieval.get('by_context_and_position', [])})
    for position in positions:
        scores = {row['context_length']: row.get('accuracy')
                  for row in retrieval.get('by_context_and_position', []) if row['target_position'] == position}
        quality_rows.append([f'Passkey @ {position:.0%} (%)', *[scores.get(length) for length in contexts]])
    for task, data in long_context.get('tasks', {}).items():
        scores = {row['input_budget']: row.get('score') for row in data.get('by_input_budget', [])}
        label = {'qasper': 'Qasper F1', 'hotpotqa': 'HotpotQA F1', 'gov_report': 'GovReport ROUGE-L'}.get(task, task)
        quality_rows.append([label, *[scores.get(length) for length in contexts]])
    if quality_rows:
        parts += ['\n## Quality by input length\n\n', table(['Task / score (0–100)', *contexts], quality_rows)]

    # One memory/diagnostic table. Values with different accounting scopes stay distinct.
    memory_rows = []
    for key, label, unit, divisor in (
        ('weights', 'GPU weights', 'GiB', 1024 ** 3),
        ('persistent_model_buffers', 'Persistent model buffers', 'MiB', 1024 ** 2),
        ('kv_cache_allocated_pool', 'KV allocated pool', 'GiB', 1024 ** 3),
        ('kv_cache_metadata', 'KV GPU metadata', 'MiB', 1024 ** 2),
        ('execution_input_buffers', 'Execution input buffers', 'MiB', 1024 ** 2),
        ('unattributed_live_allocator_allocations', 'Unattributed live allocations', 'MiB', 1024 ** 2),
    ):
        if key in components:
            memory_rows.append([label, scaled(components[key].get('bytes'), divisor), unit])
    memory_rows += [
        ['KV occupied pages (peak)', scaled(value(runtime.get('live_kv_occupancy', {}), 'peak_bytes'), 1024 ** 2), 'MiB'],
        ['Temporary PyTorch allocations (peak)', scaled(value(runtime.get('temporary_workspaces', {}), 'peak_bytes'), 1024), 'KiB'],
        ['Sampled peak GPU memory', value(metrics.get('peak_memory', {}), 'value'), 'GiB'],
    ]
    for key, label in (('weight_tensor_storage', 'Weight compression'), ('kv_cache', 'KV compression'),
                       ('total_runtime_footprint', 'Resident runtime ratio'),
                       ('active_runtime_tensor_footprint', 'Active runtime tensor ratio')):
        data = metrics.get('compression_ratio', {}).get('components', {}).get(key, {})
        memory_rows.append([label, value(data, 'value'), '×'])
    context = metrics.get('max_context', {})
    maximum = value(context, 'max_total_tokens')
    if maximum is not None and str(context.get('interpretation', '')).startswith('verified_lower_bound'):
        maximum = f'≥{maximum:,}'
    memory_rows.append(['Memory-feasible context (input + output)', maximum, 'tokens'])
    overhead = metrics.get('adaptive_overhead', {})
    memory_rows.append(['Adaptive overhead', 'N/A' if overhead.get('status') == 'not_applicable'
                        else overhead.get('status', 'pending'), ''])
    for phase, measurement in overhead.get('phases', {}).items():
        if 'stages' in measurement:
            for stage, interval in measurement['stages'].items():
                if interval.get('status') == 'ok':
                    memory_rows.append([f'{phase.title()} {stage.replace("_", " ")}',
                                        interval.get('ms_per_processed_token'), 'ms/token'])
        else:
            memory_rows.append([f'{phase.title()} adaptive full-path delta',
                                measurement.get('delta_ms_per_processed_token'), 'ms/token'])
    parts += ['\n## Memory and compression\n\n', table(['Measurement', method, 'Unit'], memory_rows)]

    # Consolidated values already calculated from all repetitions at this load.
    prompt_lengths = sorted({row['input_tokens'] for row in metrics.get('ttft', {}).get('by_workload', [])
                             if row.get('output_tokens') == 128 and row.get('concurrency') == 1
                             and row.get('request_rate') is None})
    if prompt_lengths:
        rows = []
        for key, label in (('ttft', 'TTFT p50 / p95 (ms)'), ('tpot', 'TPOT p50 / p95 (ms/token)'),
                           ('e2e_latency', 'End-to-end p50 / p95 (ms)')):
            rows.append([label, *[latency_pair(serving_row(metrics.get(key, {}), length)) for length in prompt_lengths]])
        for key, field, label in (
            ('throughput', 'output_tokens_per_second', 'Output tokens/s'),
            ('throughput', 'completed_requests_per_second', 'Completed requests/s'),
            ('goodput', 'slo_compliant_requests_per_second', 'Goodput (requests/s)'),
        ):
            rows.append([label, *[serving_row(metrics.get(key, {}), length).get(field) for length in prompt_lengths]])
        capacities = {row['input_tokens']: row for row in metrics.get('max_concurrency', {}).get('by_workload', [])
                      if row.get('output_tokens') == 128}
        def capacity(length):
            row = capacities.get(length, {})
            amount = row.get('max_sustainable_concurrency')
            if amount is not None and row.get('interpretation') == 'at_least_highest_tested_level':
                return f'≥{amount}'
            return amount
        rows.append(['Max sustainable concurrency (tested grid)', *[capacity(length) for length in prompt_lengths]])
        parts += ['\n## Serving\n\n',
                  'Latency/rates: concurrency 1, 128 output tokens; all repetitions combined.\n\n',
                  table(['Measurement', *[f'{length:,} input tokens' for length in prompt_lengths]], rows)]
        limits = metrics.get('goodput', {}).get('slo_thresholds', {})
        parts += [f"\nGoodput SLO: TTFT ≤ {cell(limits.get('ttft_ms'))} ms; TPOT ≤ {cell(limits.get('tpot_ms'))} ms/token.\n"]

    kernel = metrics.get('kernel_time', {})
    events, trace = kernel.get('cuda_events', {}), kernel.get('individual_kernels', {})
    kernel_rows = []
    for phase in ('prefill', 'decode'):
        event_phase = events.get('phases', {}).get(phase, {})
        trace_phase = trace.get('phases', {}).get(phase, {})
        steps = sum(worker.get(phase, {}).get('steps', 0)
                    for worker in trace.get('workload', {}).get('worker_stats', [])) or None
        kernel_rows.append([phase.title(), event_phase.get('mean'), trace_phase.get('sum_kernel_ms'), steps])
    parts += ['\n## Kernel timing (eager diagnostic)\n\n',
              table(['Phase', 'CUDA-event mean (ms/step)', 'Nsight kernel sum (ms)', 'Traced steps'], kernel_rows)]

    if baseline_only:
        parts += ['\nPrecision comparison: FP16 reference only.\n']
    elif quality.get('status') == 'ok':
        # Current checkpoint only; historical precision points stay in the JSON.
        points = [point for point in quality.get('points', []) if point.get('source') == 'current_run']
        parts += ['\n## Quality change vs FP16\n\n', table(['Score', 'Δ loss', 'Unit'], (
            [point['metric'], point.get('quality_loss'), point.get('quality_loss_unit')] for point in points))]
    issues = [[label, metrics.get(key, {}).get('status', 'pending')]
              for _, key, label in METRICS if metrics.get(key, {}).get('status') not in ('ok', 'not_applicable')]
    if issues:
        parts += ['\n## Pending / unavailable\n\n', table(['Metric', 'Status'], issues)]
    parts += ['\nFull sweeps, raw measurements and accounting limits: companion JSON.\n']
    return ''.join(parts)


def write_markdown_report(results: dict, json_path: Path) -> Path:
    path = json_path.with_suffix('.md')
    content = render_markdown(results, json_path.name)
    temporary = path.with_suffix('.md.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)
    return path
