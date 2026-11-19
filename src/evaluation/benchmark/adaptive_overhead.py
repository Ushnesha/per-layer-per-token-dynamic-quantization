"""Adaptive-stage timing contract and analysis; static paths are not applicable.

An adaptive runner declares aditya_overhead_capabilities with adaptive=True,
backend_id, attribution='stages' or 'fused', and a list of supported stages.
For stages, wrap real operations in runner.aditya_overhead_recorder.stage(name).
An optional static_reference_model names a saved matched static run explicitly.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path

from .quality_tradeoff import compatibility_reason

STAGES = ('precision_selection', 'quantization', 'dequantization', 'packing',
          'cache_reorganization', 'routing')


class CudaInterval:
    def __init__(self):
        import torch
        self.torch = torch
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
    def start(self):
        self.begin.record()
    def stop(self):
        self.torch.cuda.synchronize()
        self.end.record()
        self.end.synchronize()
        return self.begin.elapsed_time(self.end)


class OverheadRecorder:
    """Serialize nonnested stage intervals; preserve unobserved stages as missing."""

    def __init__(self, stages, timer_factory=CudaInterval):
        if not stages or any(stage not in STAGES for stage in stages):
            raise ValueError('Declare at least one recognized adaptive overhead stage.')
        self.stages = set(stages)
        self.timer_factory = timer_factory
        self.records = []
        self.current = None
        self.active_stage = False

    @contextmanager
    def step(self, phase, processed_tokens):
        if self.current is not None or phase not in ('prefill', 'decode') or type(processed_tokens) is not int or processed_tokens <= 0:
            raise ValueError('Step requires a valid phase, positive token count, and no active step.')
        record = {'phase': phase, 'processed_tokens': processed_tokens, 'stages': []}
        timer = self.timer_factory()
        timer.start()
        self.current = record
        try:
            yield
            record['elapsed_ms'] = timer.stop()
            self.records.append(record)
        finally:
            self.current = None
            self.active_stage = False

    @contextmanager
    def stage(self, name):
        if self.current is None or self.active_stage or name not in self.stages:
            raise ValueError('Stage must be declared, inside a step, and nonnested.')
        timer = self.timer_factory()
        timer.start()
        self.active_stage = True
        try:
            yield
            self.current['stages'].append({'name': name, 'elapsed_ms': timer.stop()})
        finally:
            self.active_stage = False


def summarize_stage_records(records: list[dict]) -> dict:
    if not records:
        return {'status': 'blocked', 'reason': 'No instrumented adaptive steps were collected.'}
    phases = {}
    for record in records:
        tokens, elapsed = record['processed_tokens'], record['elapsed_ms']
        if record['phase'] not in ('prefill', 'decode') or type(tokens) is not int or tokens <= 0:
            raise ValueError('Invalid adaptive step shape.')
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Adaptive step interval must be finite and positive.')
        phase = phases.setdefault(record['phase'], {'tokens': 0, 'step_ms': 0, 'stages': {}})
        stage_total = 0
        for stage in record['stages']:
            name, duration = stage['name'], stage['elapsed_ms']
            if name not in STAGES or not math.isfinite(duration) or duration < 0:
                raise ValueError('Invalid adaptive stage measurement.')
            phase['stages'][name] = phase['stages'].get(name, 0) + duration
            stage_total += duration
        if stage_total > elapsed + 1e-6 * max(1, elapsed):
            raise ValueError('Nonnested stage durations exceed the containing step interval.')
        phase['tokens'] += tokens
        phase['step_ms'] += elapsed
    output = {}
    observed = False
    for name, phase in phases.items():
        stages = {}
        for stage in STAGES:
            if stage not in phase['stages']:
                stages[stage] = {'status': 'not_observed'}
                continue
            observed = True
            duration = phase['stages'][stage]
            stages[stage] = {'status': 'ok', 'total_ms': duration,
                             'ms_per_processed_token': duration / phase['tokens'],
                             'percent_of_measured_step_interval': 100 * duration / phase['step_ms']}
        output[name] = {'processed_tokens': phase['tokens'], 'measured_step_ms': phase['step_ms'],
                        'stages': stages}
    return {'status': 'ok' if observed else 'blocked', 'phases': output,
            'reason': None if observed else 'No real adaptive stages were observed inside measured steps.'}


def compare_fused(current: dict, previous: list[dict], reference_model: str) -> dict:
    """Compare explicitly selected full paths; do not label a delta as isolated cost."""
    candidates = [run for run in previous if run.get('model') == reference_model
                  and compatibility_reason(current, run) is None]
    current_gpu = current.get('metrics', {}).get('peak_memory', {}).get('gpu_uuid')
    if (not current_gpu or not candidates
            or current.get('metrics', {}).get('peak_memory', {}).get('status') != 'ok'):
        return {'status': 'blocked', 'reason': 'An explicit compatible static reference and matching GPU identity are required.'}
    for reference in sorted(candidates, key=lambda run: run.get('timestamp', ''), reverse=True):
        peak = reference.get('metrics', {}).get('peak_memory', {})
        if peak.get('gpu_uuid') != current_gpu or peak.get('status') != 'ok':
            continue
        baseline = reference.get('metrics', {}).get('kernel_time', {}).get('cuda_events', {})
        adaptive = current.get('metrics', {}).get('kernel_time', {}).get('cuda_events', {})
        if baseline.get('status') != 'ok' or adaptive.get('status') != 'ok':
            continue
        if baseline.get('engine_settings') != adaptive.get('engine_settings') or baseline.get('workload') != adaptive.get('workload'):
            continue
        if baseline.get('adaptive_instrumentation', {}).get('status') == 'ok':
            continue  # Reference must be a static execution path.
        baseline_trials, adaptive_trials = baseline.get('trials', []), adaptive.get('trials', [])
        comparisons = {}
        for phase in ('prefill', 'decode'):
            a = [row for row in adaptive_trials if row['phase'] == phase]
            b = [row for row in baseline_trials if row['phase'] == phase]
            if not a or len(a) != len(b) or [row['processed_tokens'] for row in a] != [row['processed_tokens'] for row in b]:
                continue
            a_ms = sum(row['elapsed_ms'] for row in a)
            b_ms = sum(row['elapsed_ms'] for row in b)
            if not all(math.isfinite(value) and value >= 0 for value in (a_ms, b_ms)) or b_ms <= 0:
                raise ValueError('Invalid full-path timing measurements.')
            comparisons[phase] = {'delta_ms_per_processed_token': (a_ms - b_ms) / sum(row['processed_tokens'] for row in a),
                                  'relative_step_time_change_percent': 100 * (a_ms - b_ms) / b_ms}
        if comparisons:
            return {'status': 'ok', 'reference_model': reference_model, 'phases': comparisons,
                    'interpretation': 'Signed complete-path timing difference; not isolated precision-management cost.'}
    return {'status': 'blocked', 'reason': 'No static reference has matching GPU, workload, and timing protocol.'}


def evaluate_adaptive_overhead(current: dict, results_dir: Path) -> dict:
    events = current.get('metrics', {}).get('kernel_time', {}).get('cuda_events', {})
    instrument = events.get('adaptive_instrumentation', {})
    if instrument.get('status') != 'ok':
        return {'status': 'not_applicable', 'reason': 'This runtime exposes no adaptive operations; adaptive overhead is not applicable to FP16/static inference. An adaptive backend is not implemented by this evaluator.',
                'required_stages': list(STAGES), 'static_quantization_is_not_adaptive_overhead': True}
    capabilities = instrument['capabilities']
    if capabilities.get('adaptive') is not True or not capabilities.get('backend_id'):
        raise ValueError('Overhead requires explicit adaptive backend identity.')
    records = [{**record, 'trial': trial['trial']} for trial in instrument.get('trials', [])
               for record in trial['records']]
    attribution = capabilities['attribution']
    if attribution == 'stages':
        measurement = summarize_stage_records(records)
    elif attribution == 'fused':
        previous = []
        for file in sorted(results_dir.glob('evaluation_*.json')):
            try:
                run = json.loads(file.read_text())
                if isinstance(run, dict):
                    previous.append(run)
            except (OSError, ValueError):
                continue
        measurement = compare_fused(current, previous, capabilities.get('static_reference_model'))
    else:
        raise ValueError('Adaptive attribution must be stages or fused.')
    return {
        **measurement, 'backend_id': capabilities['backend_id'], 'attribution': attribution,
        'records': records,
        'management_step_ablations': {'status': 'blocked',
                                     'reason': 'Disabled-stage variants require explicit adaptive backend support.'},
        'limitations': [
            'Stage CUDA-event intervals include boundary/launch gaps and are not pure GPU-kernel busy time.',
            'Synchronization used for stage attribution can perturb execution; do not substitute these values for serving latency.',
            'Unobserved stages are not reported as zero; fused execution requires matched full-path comparison.',
            'Individual-stage disabled ablations require actual backend support and are not synthesized.',
        ],
    }
