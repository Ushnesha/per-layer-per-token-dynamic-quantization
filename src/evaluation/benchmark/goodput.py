"""Successfully completed requests meeting prespecified TTFT and TPOT SLOs."""

import math

from .throughput import serving_trials, trial_throughput
from .tpot import request_tpot

DEFAULT_SLOS = {'ttft_ms': 1000.0, 'tpot_ms': 50.0}
BOUNDARY_TOLERANCE_MS = 1e-6


def validate_slos(limits: dict) -> dict:
    if set(limits) != {'ttft_ms', 'tpot_ms'}:
        raise ValueError('Goodput requires both TTFT and TPOT latency thresholds.')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in limits.values()):
        raise ValueError('SLO thresholds must be finite positive milliseconds.')
    return dict(limits)


def assess_request(request: dict, limits: dict) -> dict:
    limits = validate_slos(limits)
    if request.get('status') != 'ok':
        return {'status': 'failed_request', 'reason': request.get('error', 'Streaming request failed.')}
    tpot = request_tpot(request)
    if tpot['status'] != 'ok':
        return {'status': 'unassessable', 'reason': tpot['reason']}
    measurements = {'ttft_ms': request['ttft_ms'], 'tpot_ms': tpot['value']}
    violated = [name for name, value in measurements.items()
                if value > limits[name] and not math.isclose(value, limits[name], rel_tol=1e-12,
                                                             abs_tol=BOUNDARY_TOLERANCE_MS)]
    return {'status': 'slo_violation' if violated else 'slo_met',
            **measurements, 'violated_thresholds': violated}


def trial_goodput(trial: dict, limits=None) -> dict:
    limits = validate_slos(DEFAULT_SLOS if limits is None else limits)
    throughput = trial_throughput(trial)
    if throughput['status'] != 'ok':
        return throughput
    invalid = {item['source_request_index']: item['reason'] for item in throughput['invalid_request_details']}
    records = []
    for index, request in enumerate(trial['requests']):
        if index in invalid:
            assessed = {'status': 'invalid', 'reason': invalid[index]}
        else:
            try:
                assessed = assess_request(request, limits)
            except (KeyError, TypeError, ValueError) as error:
                assessed = {'status': 'unassessable', 'reason': str(error)}
        records.append({'source_request_index': index, **assessed})
    met = sum(row['status'] == 'slo_met' for row in records)
    violated = sum(row['status'] == 'slo_violation' for row in records)
    unassessable = sum(row['status'] == 'unassessable' for row in records)
    return {
        'status': 'ok', 'duration_seconds': throughput['duration_seconds'],
        'attempted_requests': throughput['attempted_requests'],
        'completed_requests': throughput['completed_requests'],
        'failed_requests': throughput['failed_requests'], 'invalid_requests': throughput['invalid_requests'],
        'slo_met_requests': met, 'slo_violating_requests': violated,
        'unassessable_requests': unassessable,
        'slo_compliant_requests_per_second': met / throughput['duration_seconds'],
        'slo_attainment_percent_all_attempts': 100 * met / throughput['attempted_requests'],
        'slo_attainment_percent_completed': 100 * met / throughput['completed_requests'] if throughput['completed_requests'] else None,
        'threshold_violation_counts': {name: sum(name in row.get('violated_thresholds', []) for row in records)
                                       for name in limits},
        'maximum_client_dispatch_lag_ms': throughput['maximum_client_dispatch_lag_ms'],
        'requests': records,
    }


def combine_goodput(trials: list[dict]) -> dict:
    if not trials:
        return {'status': 'blocked', 'reason': 'No usable goodput measurement windows.'}
    counts = {name: sum(trial[name] for trial in trials) for name in (
        'attempted_requests', 'completed_requests', 'failed_requests', 'invalid_requests',
        'slo_met_requests', 'slo_violating_requests', 'unassessable_requests')}
    duration = sum(trial['duration_seconds'] for trial in trials)
    return {
        'status': 'ok', **counts, 'duration_seconds': duration, 'measured_trials': len(trials),
        'slo_compliant_requests_per_second': counts['slo_met_requests'] / duration,
        'slo_attainment_percent_all_attempts': 100 * counts['slo_met_requests'] / counts['attempted_requests'],
        'slo_attainment_percent_completed': 100 * counts['slo_met_requests'] / counts['completed_requests'] if counts['completed_requests'] else None,
        'threshold_violation_counts': {name: sum(trial['threshold_violation_counts'][name] for trial in trials)
                                       for name in DEFAULT_SLOS},
        'maximum_client_dispatch_lag_ms': max(trial['maximum_client_dispatch_lag_ms'] for trial in trials),
    }


def evaluate_goodput(serving: dict, limits=None) -> dict:
    limits = validate_slos(DEFAULT_SLOS if limits is None else limits)
    trials, groups = [], {}
    for collection, index, trial in serving_trials(serving):
        source = {'source_collection': collection, 'source_trial_index': index}
        if trial.get('status') == 'skipped':
            trials.append({**source, 'status': 'skipped', 'reason': trial.get('reason')})
            continue
        try:
            result = trial_goodput(trial, limits)
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
                          'arrival_mode': mode, 'request_rate': rate, **combine_goodput(rows)})
    return {
        'status': 'ok' if summaries else 'blocked', 'source_metric': 'ttft',
        'reason': None if summaries else 'No usable streaming measurements with explicit duration windows.',
        'slo_thresholds': limits, 'slo_boundary_tolerance_ms': BOUNDARY_TOLERANCE_MS,
        'slo_rule': 'successful complete response AND TTFT <= threshold AND TPOT <= threshold',
        'unit': 'SLO-compliant requests/s',
        'formula': 'successful requests meeting every latency SLO / measured wall-clock seconds',
        'by_workload': summaries, 'trials': trials,
        'workload': serving.get('workload'), 'engine_settings': serving.get('engine_settings'),
        'limitations': [
            'Thresholds are prespecified research defaults, not universal production requirements.',
            'Request completion and valid token counts are required; failures and unassessable TPOT do not qualify.',
            'Zero goodput with a valid window is meaningful; missing measurements remain blocked.',
            'Startup and warm-up are excluded; time spent on failures and SLO violations remains in the denominator.',
            'Scheduled Poisson arrivals are reproducible; inspect client dispatch lag before claiming equal offered load.',
            'Adaptive serving and matched-budget comparisons remain pending.',
        ],
    }
