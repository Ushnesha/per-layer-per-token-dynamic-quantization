"""CUDA-event model-step intervals and optional Nsight Systems kernel times."""

from .worker_extension import WORKER_EXTENSION

import json
from contextlib import nullcontext
import math
import os
from pathlib import Path
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from types import MethodType

from .profiling_workload import INPUT_TOKENS, OUTPUT_TOKENS, PROFILING_SETTINGS, STEP_LIMITS, profile_artifacts

WARMUP_REQUESTS = 3
MEASURED_REQUESTS = 10


def progress(message: str) -> None:
    print(f'[{datetime.now(timezone.utc).isoformat()}] CUDA timing: {message}', flush=True)


def timing_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Timing samples must be nonempty, finite, and nonnegative.')
    return {'unit': 'ms', 'samples': len(values), 'mean': statistics.mean(values),
            'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0.0}


def install_timing(worker):
    import torch
    runner = worker.model_runner
    if hasattr(worker, '_aditya_timing'):
        raise RuntimeError('Timing instrumentation is already installed.')
    original = runner.execute_model
    capabilities = getattr(runner, 'aditya_overhead_capabilities', None)
    recorder = None
    if capabilities is not None:
        if capabilities.get('adaptive') is not True or not capabilities.get('backend_id'):
            raise ValueError('Adaptive timing capabilities must declare backend identity and adaptive=True.')
        if capabilities.get('attribution') == 'stages':
            from .adaptive_overhead import OverheadRecorder
            recorder = OverheadRecorder(capabilities.get('stages', []))
            runner.aditya_overhead_recorder = recorder
        elif capabilities.get('attribution') != 'fused':
            raise ValueError('Unknown adaptive overhead attribution mode.')
    state = {'original': original, 'prefill': [], 'decode': [], 'capabilities': capabilities, 'recorder': recorder}
    worker._aditya_timing = state
    def execute(self, scheduler_output, *args, **kwargs):
        scheduled = list(scheduler_output.num_scheduled_tokens.values())
        if not scheduled:
            return original(scheduler_output, *args, **kwargs)
        if len(scheduled) != 1 or scheduled[0] <= 0:
            raise ValueError('Timing requires one active request with positive scheduled tokens.')
        phase = 'prefill' if scheduled[0] > 1 else 'decode'
        if len(state[phase]) >= STEP_LIMITS[phase]:
            return original(scheduler_output, *args, **kwargs)
        torch.cuda.synchronize(worker.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        with recorder.step(phase, scheduled[0]) if recorder is not None else nullcontext():
            result = original(scheduler_output, *args, **kwargs)
        # Wait for all streams, then record the end marker on the current stream.
        # This is an elapsed step interval, not the sum of individual kernels.
        torch.cuda.synchronize(worker.device)
        end.record()
        end.synchronize()
        state[phase].append({'elapsed_ms': start.elapsed_time(end), 'processed_tokens': scheduled[0]})
        return result
    runner.execute_model = MethodType(execute, runner)
    return {'status': 'ok'}


def remove_timing(worker):
    state = worker._aditya_timing
    worker.model_runner.execute_model = state['original']
    if state['recorder'] is not None:
        del worker.model_runner.aditya_overhead_recorder
    del worker._aditya_timing
    return {**{phase: state[phase] for phase in STEP_LIMITS},
            'adaptive_overhead': {'capabilities': state['capabilities'],
                                  'records': state['recorder'].records if state['recorder'] is not None else []}}


def evaluate_cuda_events(model_path: str) -> dict:
    progress('importing GPU libraries')
    try:
        import torch
        if not torch.cuda.is_available():
            return {'status': 'unsupported', 'reason': 'CUDA-event timing requires an NVIDIA GPU.'}
        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except ImportError as error:
        return {'status': 'unsupported', 'reason': str(error)}
    progress('loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    body = tokenizer('Describe the history and natural features of the city. ', add_special_tokens=False)['input_ids']
    if not body or tokenizer.bos_token_id is None:
        raise ValueError('Cannot construct the kernel-timing workload.')
    tokens = [tokenizer.bos_token_id] + (body * (INPUT_TOKENS // len(body) + 1))[:INPUT_TOKENS - 1]
    progress('creating engine and loading model')
    engine = LLM(model=model_path, worker_extension_cls=WORKER_EXTENSION, **PROFILING_SETTINGS)
    progress('engine ready')
    if not callable(getattr(engine, 'collective_rpc', None)):
        return {'status': 'unsupported', 'reason': 'Worker-side timing requires vLLM collective_rpc.'}
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    prompt = [TokensPrompt(prompt_token_ids=tokens)]
    for warmup in range(WARMUP_REQUESTS):
        progress(f'warmup {warmup + 1}/{WARMUP_REQUESTS} starting')
        engine.generate(prompt, params, use_tqdm=False)
        progress(f'warmup {warmup + 1}/{WARMUP_REQUESTS} finished')
    trials = []
    overhead_trials = []
    capabilities = None
    for trial in range(MEASURED_REQUESTS):
        progress(f'trial {trial + 1}: installing timing instrumentation')
        engine.collective_rpc('aditya_install_timing', timeout=60)
        try:
            progress(f'trial {trial + 1}: measured generation starting')
            outputs = engine.generate(prompt, params, use_tqdm=False)
            progress(f'trial {trial + 1}: measured generation finished')
        finally:
            progress(f'trial {trial + 1}: collecting samples and removing instrumentation')
            stats = engine.collective_rpc('aditya_remove_timing', timeout=60)
            progress(f'trial {trial + 1}: samples collected')
        if len(stats) != 1:
            raise ValueError('Kernel timing currently requires one GPU worker.')
        overhead = stats[0].get('adaptive_overhead', {})
        if overhead.get('capabilities') is not None:
            if capabilities is not None and capabilities != overhead['capabilities']:
                raise ValueError('Adaptive capabilities changed between timing trials.')
            capabilities = overhead['capabilities']
            overhead_trials.append({'trial': trial, 'records': overhead['records']})
        if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != tokens or len(outputs[0].outputs) != 1:
            raise ValueError('Timing workload returned altered or incomplete prompts.')
        if len(outputs[0].outputs[0].token_ids) != OUTPUT_TOKENS:
            raise ValueError('Timing workload generated an unexpected output length.')
        for phase, expected in STEP_LIMITS.items():
            rows = stats[0][phase]
            if len(rows) != expected or any(row['processed_tokens'] != (INPUT_TOKENS if phase == 'prefill' else 1) for row in rows):
                raise ValueError('Unexpected prefill/decode sample count or shape.')
            trials.extend({'trial': trial, 'phase': phase, **row} for row in rows)
    progress('sample counts and output lengths validated')
    return {
        'status': 'ok', 'measurement': 'synchronized CUDA-event elapsed model-step intervals',
        'phases': {phase: timing_summary([row['elapsed_ms'] for row in trials if row['phase'] == phase])
                   for phase in STEP_LIMITS},
        'trials': trials, 'engine_settings': PROFILING_SETTINGS,
        'workload': {'input_tokens': INPUT_TOKENS, 'output_tokens': OUTPUT_TOKENS,
                     'batch_size': 1, 'warmup_requests': WARMUP_REQUESTS, 'measured_requests': MEASURED_REQUESTS,
                     'measured_steps_per_request': STEP_LIMITS},
        'scope': 'includes launch gaps and synchronization boundary effects; not individual kernel durations',
        'adaptive_instrumentation': {
            'status': 'ok' if capabilities is not None else 'blocked',
            'capabilities': capabilities, 'trials': overhead_trials,
        },
    }


def union_duration_ns(intervals: list[tuple[int, int]]) -> int:
    total, end_previous = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('Kernel end precedes its start.')
        if end_previous is None or start >= end_previous:
            total += end - start
        elif end > end_previous:
            total += end - end_previous
        end_previous = max(end, end_previous) if end_previous is not None else end
    return total


def parse_kernel_trace(database: Path) -> dict:
    """Attribute kernels to synchronized named ranges in the same worker process."""
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        strings = {row['id']: row['value'] for row in connection.execute('SELECT id, value FROM StringIds')}
        ranges = []
        for row in connection.execute('SELECT * FROM NVTX_EVENTS'):
            keys = row.keys()
            text = row['text'] if 'text' in keys and row['text'] else strings.get(row['textId']) if 'textId' in keys else None
            if text not in ('aditya_prefill', 'aditya_decode') or row['end'] is None or row['end'] < row['start']:
                continue
            ranges.append({'phase': text.removeprefix('aditya_'), 'start': row['start'], 'end': row['end'],
                           'pid': row['globalTid'] >> 24})
        kernels = []
        for row in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL'):
            start, end = row['start'], row['end']
            if end < start:
                raise ValueError('Invalid kernel timestamps.')
            pid = row['globalPid'] >> 24
            owners = [item for item in ranges if item['pid'] == pid and item['start'] <= start and end <= item['end']]
            if not owners:
                continue
            if len(owners) != 1:
                raise ValueError('Kernel belongs to overlapping profiling ranges.')
            name_field = 'demangledName' if 'demangledName' in row.keys() else 'shortName'
            name = strings.get(row[name_field])
            if name is None:
                raise ValueError('Kernel name is absent from the trace string table.')
            kernels.append({'phase': owners[0]['phase'], 'kernel': name, 'pid': pid,
                            'device': row['deviceId'], 'stream': row['streamId'],
                            'start_ns': start, 'end_ns': end, 'duration_ms': (end - start) / 1_000_000})
    if not kernels or {row['phase'] for row in kernels} != {'prefill', 'decode'}:
        raise ValueError('Trace is missing marked prefill or decode kernels.')
    if len({(row['pid'], row['device']) for row in kernels}) != 1:
        raise ValueError('Kernel trace must contain exactly one profiled GPU worker/device.')
    phases = {}
    for phase in STEP_LIMITS:
        rows = [row for row in kernels if row['phase'] == phase]
        groups = {}
        for row in rows:
            groups.setdefault(row['kernel'], []).append(row['duration_ms'])
        phases[phase] = {
            'sum_kernel_ms': sum(row['duration_ms'] for row in rows),
            'gpu_busy_ms': union_duration_ns([(row['start_ns'], row['end_ns']) for row in rows]) / 1_000_000,
            'kernel_invocations': len(rows),
            'by_kernel': [{'kernel': name, **timing_summary(values), 'total_ms': sum(values)}
                          for name, values in sorted(groups.items())],
        }
    return {'status': 'ok', 'measurement': 'Nsight Systems kernel execution intervals',
            'phases': phases, 'kernel_invocations': kernels,
            'scope': 'GPU busy time merges overlaps; summed kernel durations can exceed elapsed/busy time'}


def evaluate_kernel_trace(model_path: str) -> dict:
    nsys = shutil.which('nsys')
    if nsys is None:
        return {'status': 'unsupported', 'reason': 'Nsight Systems CLI (nsys) is not on PATH.'}
    version = subprocess.run([nsys, '--version'], capture_output=True, text=True, timeout=30)
    with profile_artifacts('kernel_trace') as temporary:
        prefix = Path(temporary) / 'timeline'
        metadata = Path(temporary) / 'workload.json'
        command = [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--wait=all',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--export=sqlite',
                   '--force-overwrite=true', '--output', str(prefix), sys.executable, '-m',
                   'src.evaluation.benchmark.profiling_workload', model_path, str(metadata)]
        log_path = Path(temporary) / 'workload.log'
        with log_path.open('w') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800,
                                         cwd=Path(__file__).resolve().parents[3])
            except subprocess.TimeoutExpired:
                return {'status': 'failed', 'reason': 'Nsight Systems exceeded 1800 seconds.', 'worker_log': str(log_path)}
        if process.returncode != 0 or not prefix.with_suffix('.sqlite').is_file() or not metadata.is_file():
            return {'status': 'unsupported', 'reason': 'Nsight Systems did not produce a usable trace.',
                    'worker_log': str(log_path), 'diagnostics': log_path.read_text(errors='replace')[-8000:]}
        try:
            result = parse_kernel_trace(prefix.with_suffix('.sqlite'))
        except (sqlite3.Error, KeyError, ValueError) as error:
            return {'status': 'unsupported', 'reason': f'Unsupported or incomplete Nsight trace: {error}', 'artifact_directory': temporary}
        result['workload'] = json.loads(metadata.read_text())
        result['profiler_version'] = (version.stdout + version.stderr).strip()
        result['artifact_directory'] = temporary
        return result


def run_event_worker(model_path: str) -> dict:
    """Exit the event engine's process before launching the separate trace engine."""
    with profile_artifacts('cuda_events') as temporary:
        destination = Path(temporary) / 'events.json'
        command = [sys.executable, '-u', '-m', 'src.evaluation.benchmark.kernel_time', model_path, str(destination)]
        log_dir = Path(__file__).resolve().parents[3] / 'logs' / 'evaluation_workers' / os.environ.get('SLURM_JOB_ID', 'local')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / 'cuda_events.log'
        # Redirect directly to disk: buffered capture loses all progress when
        # Slurm kills the parent before subprocess.run returns.
        with log_path.open('w') as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800,
                                         cwd=Path(__file__).resolve().parents[3])
            except subprocess.TimeoutExpired:
                return {'status': 'failed', 'reason': 'CUDA-event worker exceeded 1800 seconds.',
                        'worker_log': str(log_path), 'diagnostics': log_path.read_text(errors='replace')[-20000:]}
        if process.returncode != 0 or not destination.is_file():
            return {'status': 'failed', 'reason': 'CUDA-event worker failed.',
                    'worker_log': str(log_path), 'diagnostics': log_path.read_text(errors='replace')[-20000:]}
        result = json.loads(destination.read_text())
        result['worker_log'] = str(log_path)
        if result.get('status') == 'failed':
            result['diagnostics'] = log_path.read_text(errors='replace')[-20000:]
        return result


def evaluate_kernel_time(model_path: str) -> dict:
    events = run_event_worker(model_path)
    trace = (evaluate_kernel_trace(model_path))
    statuses = [events['status'], trace['status']]
    result = {
        'status': 'ok' if all(status == 'ok' for status in statuses) else 'partial' if 'ok' in statuses else 'failed' if 'failed' in statuses else 'unsupported',
        'cuda_events': events, 'individual_kernels': trace,
        'coverage': 'inspect both measurement statuses; step intervals and kernel durations are distinct',
        'limitations': [
            'Diagnostic eager execution disables CUDA graphs, asynchronous scheduling, and chunked prefill.',
            'Only the first prefill and first three decode model steps of each request are measured.',
            'CUDA-event intervals include launch gaps; do not report them as individual attention/matmul kernel durations.',
            'Nsight traces cover one request, while event intervals use repeated requests; do not combine the two samples.',
            'Operations outside model_runner.execute_model are excluded; fused kernels remain indivisible.',
            'Matched adaptive/static comparisons require equivalent real kernels on the same hardware and workload.',
        ],
    }
    if result['status'] == 'failed':
        result['error'] = '; '.join(
            measurement.get('error') or measurement.get('reason') or 'Missing timing results'
            for measurement in (events, trace) if measurement.get('status') == 'failed')
    return result


if __name__ == '__main__':
    # Use the canonical module, as the worker extension imports its hooks here.
    from src.evaluation.benchmark.kernel_time import evaluate_cuda_events as run_events
    model_path, destination = sys.argv[1:]
    try:
        result = run_events(model_path)
    except Exception as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
                  'traceback': traceback.format_exc()}
    with Path(destination).open('w') as handle:
        json.dump(result, handle, allow_nan=False)
