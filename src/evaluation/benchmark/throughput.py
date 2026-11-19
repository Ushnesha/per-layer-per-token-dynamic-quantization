"""Useful serving throughput over explicit warm-up-free measurement windows."""

import math

from .e2e_latency import request_latency


def serving_trials(serving: dict):
    """Keep rate-controlled arrivals separate from closed-loop concurrency trials."""
    for collection in ('trials', 'request_rate_trials'):
        for index, trial in enumerate(serving.get(collection, [])):
            yield collection, index, trial


def trial_throughput(trial: dict) -> dict:
    window = trial.get('measurement_window')
    if not window:
        return {'status': 'blocked', 'reason': 'Explicit benchmark measurement window missing; no rate is inferred from latency.'}
    start, end = window['start_seconds'], window['end_seconds']
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (start, end)) or end <= start:
        raise ValueError('Benchmark window must have finite timestamps and positive duration.')
    completed, tokens, failed, invalid = 0, 0, 0, []
    requests = trial.get('requests', [])
    if not requests:
        return {'status': 'blocked', 'reason': 'No requests were attempted in the measured window.'}
    for index, request in enumerate(requests):
        if request.get('status') != 'ok':
            failed += 1
            continue
        try:
            request_latency(request)
            finished = request.get('stream_finished_seconds', request['completion_seconds'])
            if request['dispatch_seconds'] < start - 1e-6 or finished > end + 1e-6:
                raise ValueError('Successful request lies outside the measured benchmark window.')
            if request['input_tokens'] != trial['input_tokens'] or request['output_tokens'] != trial['output_tokens']:
                raise ValueError('Request token lengths differ from the workload.')
        except (KeyError, TypeError, ValueError) as error:
            invalid.append({'source_request_index': index, 'reason': str(error)})
            continue
        completed += 1
        tokens += request['output_tokens']
    duration = end - start
    return {
        'status': 'ok', 'duration_seconds': duration, 'attempted_requests': len(requests),
        'completed_requests': completed, 'failed_requests': failed, 'invalid_requests': len(invalid),
        'completed_output_tokens': tokens, 'output_tokens_per_second': tokens / duration,
        'completed_requests_per_second': completed / duration,
        'invalid_request_details': invalid, 'measurement_window': window,
        'maximum_client_dispatch_lag_ms': max((row.get('client_dispatch_lag_ms', 0) for row in requests), default=0),
    }


def combine_trials(trials: list[dict]) -> dict:
    if not trials:
        return {'status': 'blocked', 'reason': 'No valid measured trial windows.'}
    duration = sum(trial['duration_seconds'] for trial in trials)
    completed = sum(trial['completed_requests'] for trial in trials)
    tokens = sum(trial['completed_output_tokens'] for trial in trials)
    return {
        'status': 'ok', 'measured_trials': len(trials), 'duration_seconds': duration,
        'completed_requests': completed, 'completed_output_tokens': tokens,
        'attempted_requests': sum(trial['attempted_requests'] for trial in trials),
        'failed_requests': sum(trial['failed_requests'] for trial in trials),
        'invalid_requests': sum(trial['invalid_requests'] for trial in trials),
        'output_tokens_per_second': tokens / duration,
        'completed_requests_per_second': completed / duration,
        'maximum_client_dispatch_lag_ms': max(trial['maximum_client_dispatch_lag_ms'] for trial in trials),
        'aggregation': 'sum of completed work / sum of measured durations; not mean of trial rates',
    }


def evaluate_throughput(serving: dict) -> dict:
    trials, groups = [], {}
    for collection, index, trial in serving_trials(serving):
        source = {'source_collection': collection, 'source_trial_index': index}
        if trial.get('status') == 'skipped':
            trials.append({**source, 'status': 'skipped', 'reason': trial.get('reason')})
            continue
        try:
            result = trial_throughput(trial)
        except (KeyError, TypeError, ValueError) as error:
            result = {'status': 'invalid', 'reason': str(error)}
        workload = {key: trial.get(key) for key in ('input_tokens', 'output_tokens', 'concurrency', 'request_rate')}
        workload['arrival_mode'] = trial.get('arrival_mode', 'closed_loop')
        entry = {**source, **workload, 'repetition': trial.get('repetition'), **result}
        trials.append(entry)
        if result['status'] == 'ok':
            key = (workload['input_tokens'], workload['output_tokens'], workload['concurrency'],
                   workload['arrival_mode'], workload['request_rate'])
            groups.setdefault(key, []).append(entry)
    summaries = []
    for (length, output, concurrency, mode, rate), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        summaries.append({'input_tokens': length, 'output_tokens': output, 'concurrency': concurrency,
                          'arrival_mode': mode, 'request_rate': rate, **combine_trials(rows)})
    return {
        'status': 'ok' if summaries else 'blocked', 'source_metric': 'ttft',
        'reason': None if summaries else 'No explicit usable serving measurement windows.',
        'by_workload': summaries, 'trials': trials,
        'workload': serving.get('workload'), 'engine_settings': serving.get('engine_settings'),
        'formula': 'successfully completed output tokens or requests / measured wall-clock seconds',
        'limitations': [
            'Measurement windows exclude startup and warm-up; failed requests still consume denominator time.',
            'Partial output from failed/invalid requests does not count as useful completed output.',
            'Closed-loop concurrency and seeded Poisson arrival-rate trials are reported separately.',
            'For Poisson trials, concurrency denotes the client-pool limit rather than a constant active request count.',
            'Check dispatch lag before treating scheduled request rates as achieved offered rates.',
            'Throughput is not SLO goodput and does not by itself establish acceptable model quality.',
            'Adaptive serving integration and matched GPU-budget comparisons remain pending.',
        ],
    }
