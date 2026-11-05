"""Per-request TPOT derived from the existing streaming serving measurements."""

import math

from .serving import percentile


def request_tpot(request: dict) -> dict:
    if request.get('status') != 'ok':
        return {'status': 'failed_request', 'reason': request.get('error', 'Source streaming request failed.')}
    count = request.get('output_tokens')
    arrivals = request.get('token_arrival_offsets_seconds', [])
    sizes = request.get('token_chunk_sizes', [])
    if type(count) is not int or count <= 0:
        raise ValueError('A positive integer output-token count is required.')
    if not arrivals or len(arrivals) != len(sizes):
        raise ValueError('Token arrival offsets and chunk sizes must be present and aligned.')
    if any(type(size) is not int or size <= 0 for size in sizes) or sum(sizes) != count:
        raise ValueError('Streamed chunk token counts do not match total output tokens.')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in arrivals):
        raise ValueError('Arrival offsets must be finite nonnegative seconds.')
    if any(later < earlier for earlier, later in zip(arrivals, arrivals[1:])):
        raise ValueError('Token arrival times must not decrease.')
    if not math.isclose(request['ttft_ms'], arrivals[0] * 1000, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError('Recorded TTFT differs from first-token arrival.')
    if count == 1:
        return {'status': 'not_applicable', 'reason': 'One-token outputs have no post-first-token interval.'}
    if sizes[0] != 1:
        return {'status': 'unsupported', 'reason': 'First chunk contains multiple tokens; individual first-token timing is unresolved.'}
    return {
        'status': 'ok', 'value': (arrivals[-1] - arrivals[0]) * 1000 / (count - 1),
        'unit': 'ms/output token', 'output_tokens': count,
        'post_first_token_ms': (arrivals[-1] - arrivals[0]) * 1000,
        'coalesced_later_chunks': any(size > 1 for size in sizes[1:]),
    }


def summarize_tpot(records: list[dict]) -> dict:
    values = [row['value'] for row in records if row['status'] == 'ok']
    result = {
        'attempted': len(records), 'successful': len(values),
        'failed_requests': sum(row['status'] == 'failed_request' for row in records),
        'excluded_requests': sum(row['status'] not in ('ok', 'failed_request') for row in records),
    }
    if not values:
        return {'status': 'blocked', **result, 'reason': 'No eligible multi-token streaming requests.'}
    return {'status': 'ok', **result, 'unit': 'ms/output token',
            'p50': percentile(values, 50), 'p95': percentile(values, 95),
            'mean': sum(values) / len(values),
            'percentile_method': 'linear interpolation over per-request TPOT; successful eligible requests only'}


def evaluate_tpot(ttft: dict) -> dict:
    if ttft.get('status') != 'ok' or not ttft.get('trials'):
        return {'status': 'blocked', 'reason': 'Successful streaming serving measurements are required.',
                'source_metric': 'ttft'}
    trials, groups = [], {}
    for index, trial in enumerate(ttft['trials']):
        if trial.get('status') == 'skipped':
            trials.append({'source_trial_index': index, 'status': 'skipped', 'reason': trial.get('reason')})
            continue
        records = []
        for request_index, request in enumerate(trial.get('requests', [])):
            try:
                record = request_tpot(request)
            except (KeyError, ValueError, TypeError) as error:
                record = {'status': 'invalid', 'reason': str(error)}
            records.append({'source_request_index': request_index, **record})
        summary = summarize_tpot(records)
        workload = {'input_tokens': trial['input_tokens'], 'output_tokens': trial['output_tokens'],
                    'concurrency': trial['concurrency']}
        trials.append({'source_trial_index': index, 'repetition': trial['repetition'],
                       **workload, **summary, 'requests': records})
        group = (workload['input_tokens'], workload['output_tokens'], workload['concurrency'])
        groups.setdefault(group, []).extend(records)
    summaries = [{'input_tokens': length, 'output_tokens': output, 'concurrency': concurrency,
                  **summarize_tpot(records)}
                 for (length, output, concurrency), records in sorted(groups.items())]
    return {
        'status': 'ok' if any(row['status'] == 'ok' for row in summaries) else 'blocked',
        'source_metric': 'ttft', 'unit': 'ms/output token',
        'formula': '(last_token_arrival - first_token_arrival) * 1000 / (output_tokens - 1)',
        'by_workload': summaries, 'trials': trials,
        'workload': ttft.get('workload'), 'engine_settings': ttft.get('engine_settings'),
        'limitations': [
            'Measures client-observed average decode delivery delay, not isolated GPU kernel time.',
            'Stream completion/usage/[DONE] delays are excluded; the endpoint is the last generated-token event.',
            'First chunks with multiple tokens are excluded; later coalesced chunks are flagged.',
            'Warm-up requests are absent from the source measured trials; failures and exclusions are retained.',
            'Closed-loop load and adaptive-serving integration limitations of the source benchmark still apply.',
        ],
    }
