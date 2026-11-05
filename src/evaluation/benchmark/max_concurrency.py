"""Largest tested concurrency meeting every repetition's success/SLO/memory rules."""

import math

from .goodput import DEFAULT_SLOS, trial_goodput

MIN_SUCCESS_RATE = 0.99
MIN_SLO_ATTAINMENT = 0.95


def observed_concurrency(requests: list[dict]) -> int:
    events = []
    for row in requests:
        if row.get('status') != 'ok':
            continue
        start = row['dispatch_seconds']
        end = row.get('stream_finished_seconds', row['completion_seconds'])
        if not all(math.isfinite(value) for value in (start, end)) or end <= start:
            raise ValueError('Successful request has invalid overlap timestamps.')
        events.extend([(start, 1), (end, -1)])
    active, peak = 0, 0
    for _, change in sorted(events):  # End before start on ties: [dispatch, completion).
        active += change
        peak = max(peak, active)
    return peak


def assess_trial(trial: dict, expected_requests: int) -> dict:
    goodput = trial_goodput(trial)
    if goodput['status'] != 'ok':
        return {'verdict': 'inconclusive', 'reason': goodput.get('reason')}
    if goodput['attempted_requests'] != expected_requests:
        return {'verdict': 'inconclusive', 'reason': 'Incomplete measured request count.'}
    if goodput['invalid_requests'] or goodput['unassessable_requests']:
        return {'verdict': 'inconclusive', 'reason': 'Invalid or unresolved SLO measurements.'}
    success = goodput['completed_requests'] / goodput['attempted_requests']
    attainment = goodput['slo_met_requests'] / goodput['attempted_requests']
    peak = observed_concurrency(trial['requests'])
    checks = {'success_rate': success >= MIN_SUCCESS_RATE,
              'slo_attainment': attainment >= MIN_SLO_ATTAINMENT,
              'requested_concurrency_reached': peak == trial['concurrency']}
    memory = trial.get('memory', {})
    if memory.get('status') == 'ok' and isinstance(memory.get('within_budget'), bool):
        checks['memory_budget'] = memory['within_budget']
    else:
        checks['memory_budget'] = None
    # Definite failures are retained even if another measurement is unavailable.
    if not checks['success_rate'] or not checks['slo_attainment'] or checks['memory_budget'] is False:
        verdict = 'fail'
    elif not checks['requested_concurrency_reached'] or checks['memory_budget'] is None:
        verdict = 'inconclusive'
    else:
        verdict = 'pass'
    return {'verdict': verdict, 'checks': checks, 'success_rate': success,
            'slo_attainment_all_attempts': attainment, 'observed_peak_concurrency': peak,
            'memory': memory, 'attempted_requests': goodput['attempted_requests']}


def evaluate_max_concurrency(serving: dict) -> dict:
    protocol = serving.get('workload', {})
    levels = protocol.get('concurrency_levels', [])
    repetitions = protocol.get('repetitions')
    requests_per_trial = protocol.get('measured_requests_per_trial')
    lengths = protocol.get('prompt_lengths', [])
    output = protocol.get('output_tokens')
    if (not levels or not lengths or type(repetitions) is not int or repetitions <= 0
            or type(requests_per_trial) is not int or requests_per_trial <= 0 or type(output) is not int or output <= 0):
        return {'status': 'blocked', 'reason': 'Complete concurrency sweep configuration and serving records are required.'}
    if any(type(level) is not int or level <= 0 for level in levels) or len(set(levels)) != len(levels):
        raise ValueError('Concurrency levels must be distinct positive integers.')
    results = []
    for length in lengths:
        level_results = []
        for level in sorted(levels):
            matching = [(index, row) for index, row in enumerate(serving.get('trials', []))
                        if row.get('input_tokens') == length and row.get('concurrency') == level
                        and row.get('arrival_mode', 'closed_loop') == 'closed_loop']
            ids = [row.get('repetition') for _, row in matching]
            complete = len(ids) == repetitions and set(ids) == set(range(repetitions))
            assessments = []
            for index, trial in matching:
                if trial.get('status') == 'skipped' or trial.get('output_tokens') != output:
                    assessed = {'verdict': 'inconclusive', 'reason': trial.get('reason', 'Output length differs from the sweep.')}
                else:
                    try:
                        assessed = assess_trial(trial, requests_per_trial)
                    except (KeyError, TypeError, ValueError) as error:
                        assessed = {'verdict': 'inconclusive', 'reason': str(error)}
                assessments.append({'source_trial_index': index, 'repetition': trial.get('repetition'), **assessed})
            verdicts = [row['verdict'] for row in assessments]
            verdict = ('fail' if 'fail' in verdicts else 'pass'
                       if complete and verdicts and all(value == 'pass' for value in verdicts) else 'inconclusive')
            budgets = {(row['memory'].get('gpu_uuid'), row['memory'].get('memory_budget_bytes'))
                       for row in assessments if row.get('memory', {}).get('status') == 'ok'}
            if verdict == 'pass' and (len(budgets) != 1 or any(value is None for pair in budgets for value in pair)):
                verdict = 'inconclusive'
            level_results.append({'concurrency': level, 'verdict': verdict,
                                  'all_repetitions_present': complete, 'trials': assessments})
        sweep_budgets = {(trial['memory'].get('gpu_uuid'), trial['memory'].get('memory_budget_bytes'))
                         for row in level_results for trial in row['trials']
                         if trial.get('memory', {}).get('status') == 'ok'}
        if len(sweep_budgets) > 1:
            for row in level_results:
                if row['verdict'] == 'pass':
                    row.update(verdict='inconclusive', reason='GPU identity or memory budget changed within the sweep.')
        passed = [row['concurrency'] for row in level_results if row['verdict'] == 'pass']
        maximum = max(passed) if passed else None
        if maximum is None:
            interpretation = 'no_qualified_level' if all(row['verdict'] == 'fail' for row in level_results) else 'incomplete_evidence'
        elif maximum == max(levels):
            interpretation = 'at_least_highest_tested_level'
        elif any(row['verdict'] == 'inconclusive' and row['concurrency'] > maximum for row in level_results):
            interpretation = 'lower_bound_with_incomplete_higher_tests'
        else:
            interpretation = 'largest_passing_tested_level; intermediate levels were not searched'
        results.append({'input_tokens': length, 'output_tokens': output,
                        'max_sustainable_concurrency': maximum, 'interpretation': interpretation,
                        'levels': level_results,
                        'nonmonotonic_outcomes': any(a['verdict'] == 'fail' and b['verdict'] == 'pass'
                                                   for i, a in enumerate(level_results) for b in level_results[i + 1:])})
    any_evidence = any(level['verdict'] != 'inconclusive' for row in results for level in row['levels'])
    return {
        'status': 'ok' if any_evidence else 'blocked', 'source_metric': 'ttft', 'unit': 'concurrent requests',
        'by_workload': results, 'minimum_success_rate': MIN_SUCCESS_RATE,
        'minimum_slo_attainment_all_attempts': MIN_SLO_ATTAINMENT,
        'slo_thresholds': dict(DEFAULT_SLOS), 'required_repetitions': repetitions,
        'memory_rule': 'every repetition must have an uncontaminated sampled device-growth peak within its recorded fixed budget',
        'limitations': [
            'Reports the largest passing tested client concurrency, not an unbounded system maximum.',
            'Only closed-loop trials qualify; Poisson client-pool limits do not establish simultaneous load.',
            'Every repetition must pass; pooled averages cannot hide an unstable repetition.',
            'Memory is sampled relative to a pre-server baseline and may miss short-lived peaks.',
            'Client request overlap does not establish the number of requests simultaneously executing GPU kernels.',
            'Adaptive serving integration and comparisons across different memory budgets remain pending.',
        ],
    }
