"""End-to-end complete-response latency derived from streaming request records."""

import math

from .serving import percentile
from .tpot import request_tpot


def request_latency(request: dict) -> dict:
    if request.get('status') != 'ok':
        return {'status': 'failed_request', 'reason': request.get('error', 'Source streaming request failed.')}
    # Reuse token-count/arrival validation. Single-token and coalesced-first
    # responses remain eligible here even when TPOT cannot be resolved.
    request_tpot(request)
    started = request['dispatch_seconds']
    completed = request.get('stream_finished_seconds', request['completion_seconds'])
    last_token = request['last_token_seconds']
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (started, completed, last_token)):
        raise ValueError('Request timestamps must be finite numeric seconds.')
    if not started <= last_token <= completed:
        raise ValueError('Dispatch, last-token, and completion timestamps are out of order.')
    expected_last = started + request['token_arrival_offsets_seconds'][-1]
    if not math.isclose(last_token, expected_last, rel_tol=0, abs_tol=1e-6):
        raise ValueError('Last-token timestamp differs from the streamed arrival record.')
    elapsed_ms = (completed - started) * 1000
    if not math.isfinite(elapsed_ms):
        raise ValueError('Request latency is not finite.')
    return {
        'status': 'ok', 'value': elapsed_ms, 'unit': 'ms/request',
        'last_token_latency_ms': (last_token - started) * 1000,
        'stream_tail_ms': (completed - last_token) * 1000,
        'completion_boundary': 'SSE_DONE_received' if 'stream_finished_seconds' in request else 'legacy_after_stream_validation',
    }


def summarize_latency(records: list[dict]) -> dict:
    values = [row['value'] for row in records if row['status'] == 'ok']
    counts = {'attempted': len(records), 'successful': len(values),
              'failed_requests': sum(row['status'] == 'failed_request' for row in records),
              'excluded_requests': sum(row['status'] not in ('ok', 'failed_request') for row in records)}
    if not values:
        return {'status': 'blocked', **counts, 'reason': 'No complete valid responses.'}
    return {'status': 'ok', **counts, 'unit': 'ms/request',
            'p50': percentile(values, 50), 'p95': percentile(values, 95),
            'mean': sum(values) / len(values),
            'percentile_method': 'linear interpolation over complete successful requests'}


def evaluate_e2e_latency(serving: dict) -> dict:
    if serving.get('status') != 'ok' or not serving.get('trials'):
        return {'status': 'blocked', 'source_metric': 'ttft',
                'reason': 'Successful streaming serving measurements are required.'}
    trials, groups = [], {}
    for index, trial in enumerate(serving['trials']):
        if trial.get('status') == 'skipped':
            trials.append({'source_trial_index': index, 'status': 'skipped', 'reason': trial.get('reason')})
            continue
        records = []
        for request_index, request in enumerate(trial.get('requests', [])):
            try:
                result = request_latency(request)
            except (KeyError, TypeError, ValueError) as error:
                result = {'status': 'invalid', 'reason': str(error)}
            records.append({'source_request_index': request_index, **result})
        workload = {'input_tokens': trial['input_tokens'], 'output_tokens': trial['output_tokens'],
                    'concurrency': trial['concurrency']}
        trials.append({'source_trial_index': index, 'repetition': trial['repetition'],
                       **workload, **summarize_latency(records), 'requests': records})
        key = (workload['input_tokens'], workload['output_tokens'], workload['concurrency'])
        groups.setdefault(key, []).extend(records)
    summaries = [{'input_tokens': length, 'output_tokens': output, 'concurrency': concurrency,
                  **summarize_latency(records)} for (length, output, concurrency), records in sorted(groups.items())]
    return {
        'status': 'ok' if any(row['status'] == 'ok' for row in summaries) else 'blocked',
        'source_metric': 'ttft', 'unit': 'ms/request',
        'formula': '(complete_stream_received - request_dispatch) * 1000',
        'by_workload': summaries, 'trials': trials,
        'workload': serving.get('workload'), 'engine_settings': serving.get('engine_settings'),
        'limitations': [
            'Includes loopback transport, server queueing, generation, and final stream delivery; excludes client executor waiting.',
            'Complete-response latency includes usage/[DONE] delivery; last-token latency and stream-tail delay are reported separately.',
            'Warm-up is excluded; failed and invalid requests are retained separately from latency percentiles.',
            'Closed-loop concurrency does not impose equal offered request rates across model configurations.',
            'Adaptive serving integration and matched-budget comparisons remain pending.',
        ],
    }
