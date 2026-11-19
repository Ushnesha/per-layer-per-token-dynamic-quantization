"""Sample total NVIDIA device memory around a fixed vLLM inference workload."""

import hashlib
import json
import os
import threading
import time

from .perplexity import ENGINE_SETTINGS

SAMPLE_INTERVAL_SECONDS = 0.020
INPUT_TOKENS = 512
OUTPUT_TOKENS = 128
WARMUP_REQUESTS = 3
MEASURED_REQUESTS = 3
GIB = 1024 ** 3


class MemorySampler:
    """Sample device-wide usage and keep phase boundaries and failures explicit."""

    def __init__(self, read, interval=SAMPLE_INTERVAL_SECONDS):
        self.read = read
        self.interval = interval
        self.phase = 'startup'
        self.samples = []
        self.errors = []
        self.lock = threading.Lock()
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.started = time.perf_counter()

    def sample(self):
        with self.lock:
            measurement = self.read()
            used, total = measurement['used_bytes'], measurement['total_bytes']
            if not 0 <= used <= total or total <= 0:
                raise ValueError('Invalid NVML memory measurement.')
            self.samples.append({'seconds': time.perf_counter() - self.started,
                                 'phase': self.phase, **measurement})

    def _poll(self):
        while not self.finished.wait(self.interval):
            try:
                self.sample()
            except Exception as error:
                self.errors.append(str(error))
                break

    def start(self):
        self.sample()
        self.thread.start()

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase
        self.sample()

    def stop(self):
        self.finished.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Memory sampling thread did not stop.')
        if self.errors:
            raise RuntimeError('Memory monitoring failed: ' + '; '.join(self.errors))
        self.sample()


def summarize_samples(samples: list[dict], baseline_bytes: int) -> dict:
    if not samples or not any(sample['phase'] == 'inference' for sample in samples):
        raise ValueError('No inference memory samples were collected.')
    phases = {}
    for sample in samples:
        if not 0 <= sample['used_bytes'] <= sample['total_bytes']:
            raise ValueError('Invalid memory sample.')
        phases.setdefault(sample['phase'], []).append(sample)
    peak = max(sample['used_bytes'] for sample in samples)
    inference_peak = max(sample['used_bytes'] for sample in phases['inference'])
    foreign = sorted({pid for sample in samples for pid in sample['foreign_pids']})
    times = sorted(sample['seconds'] for sample in samples)
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    return {
        'status': 'blocked' if foreign else 'ok',
        'reason': 'Unrelated GPU processes contaminated the device measurement.' if foreign else None,
        'value': peak / GIB, 'unit': 'GiB',
        'peak_device_bytes': peak,
        'inference_peak_gib': inference_peak / GIB,
        'baseline_device_gib': baseline_bytes / GIB,
        'peak_increase_over_baseline_gib': max(0, peak - baseline_bytes) / GIB,
        'phase_peaks_gib': {phase: max(row['used_bytes'] for row in rows) / GIB
                            for phase, rows in phases.items()},
        'foreign_gpu_pids': foreign, 'num_samples': len(samples),
        'maximum_observed_sampling_gap_ms': max(gaps, default=0) * 1000,
    }


def evaluate_peak_memory(model_path: str, *, include_components: bool = False) -> dict:
    # NVML includes the allocations in vLLM child workers, unlike parent-only
    # torch.cuda.max_memory_allocated(). Resolve CUDA device zero by UUID so
    # Slurm's CUDA_VISIBLE_DEVICES remapping cannot select the wrong NVML GPU.
    try:
        import pynvml as nvml
        import psutil
        import torch
    except ImportError as error:
        return {'status': 'unsupported', 'reason': str(error)}
    if not torch.cuda.is_available():
        return {'status': 'unsupported', 'reason': 'An NVIDIA CUDA device is required.'}
    uuid = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if uuid is None:
        return {'status': 'unsupported', 'reason': 'CUDA device UUID unavailable; refusing to guess the NVML device index.'}
    uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
    if uuid.startswith('MIG-'):
        return {'status': 'unsupported', 'reason': 'MIG memory accounting is not supported by this single-GPU metric.'}
    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        return {'status': 'unsupported', 'reason': f'NVML unavailable: {error}'}
    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(uuid)
        parent = psutil.Process(os.getpid())
        def read():
            info = nvml.nvmlDeviceGetMemoryInfo(handle)
            own_pids = {parent.pid, *(child.pid for child in parent.children(recursive=True))}
            processes = [*nvml.nvmlDeviceGetComputeRunningProcesses(handle),
                         *nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
            return {'used_bytes': int(info.used), 'total_bytes': int(info.total),
                    'foreign_pids': sorted({process.pid for process in processes if process.pid not in own_pids})}
        baseline = read()
        if baseline['foreign_pids']:
            return {'status': 'blocked', 'reason': 'Use an exclusive GPU allocation; unrelated processes are already active.',
                    'foreign_gpu_pids': baseline['foreign_pids'], 'gpu_uuid': uuid}

        os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        body = tokenizer('Describe the history and natural features of the city. ',
                         add_special_tokens=False)['input_ids']
        if not body or tokenizer.bos_token_id is None:
            raise ValueError('Cannot construct the fixed memory workload with this tokenizer.')
        prompt_ids = [tokenizer.bos_token_id] + (body * ((INPUT_TOKENS // len(body)) + 1))[:INPUT_TOKENS - 1]
        params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
        sampler = MemorySampler(read)
        engine_options = dict(ENGINE_SETTINGS)
        if include_components:
            from .worker_extension import WORKER_EXTENSION
            engine_options['worker_extension_cls'] = WORKER_EXTENSION
            print('Starting one engine with memory snapshot support', flush=True)
        sampler.start()
        try:
            engine = LLM(model=model_path, **engine_options)
            sampler.sample()
            sampler.set_phase('warmup')
            for _ in range(WARMUP_REQUESTS):
                engine.generate([TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False)
                sampler.sample()
            sampler.set_phase('inference')
            generated_counts = []
            for _ in range(MEASURED_REQUESTS):
                outputs = engine.generate([TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False)
                if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != prompt_ids or len(outputs[0].outputs) != 1:
                    raise ValueError('Memory workload returned incomplete or altered prompts.')
                count = len(outputs[0].outputs[0].token_ids)
                if count != OUTPUT_TOKENS:
                    raise ValueError('Memory workload did not generate the fixed output length.')
                generated_counts.append(count)
                sampler.sample()
        finally:
            sampler.stop()
        component_result = {}
        tracking = None
        if include_components:
            # Run attribution separately after the uninstrumented NVML window.
            # Synchronization/history tracing must not alter the serving/peak run.
            from .component_memory import snapshot_components
            print('Inference complete; tracing worker allocations and live KV pages', flush=True)
            installed = engine.collective_rpc('aditya_install_memory_tracking', timeout=60)
            if len(installed) != 1 or installed[0].get('status') != 'ok':
                raise RuntimeError(f'Memory instrumentation installation failed: {installed}')
            try:
                for _ in range(MEASURED_REQUESTS):
                    outputs = engine.generate([TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False)
                    if (len(outputs) != 1 or list(outputs[0].prompt_token_ids) != prompt_ids
                            or len(outputs[0].outputs) != 1
                            or len(outputs[0].outputs[0].token_ids) != OUTPUT_TOKENS):
                        raise ValueError('Instrumented memory workload returned incomplete output.')
            finally:
                collected = engine.collective_rpc('aditya_remove_memory_tracking', timeout=120)
            if len(collected) != 1 or collected[0].get('status') != 'ok':
                raise RuntimeError(f'Memory instrumentation collection failed: {collected}')
            tracking = collected[0]
            tracking['workload'] = {'input_tokens': INPUT_TOKENS, 'output_tokens': OUTPUT_TOKENS,
                'batch_size': 1, 'measured_requests': MEASURED_REQUESTS,
                'prompt_ids_sha256': hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
                'temperature': 0.0, 'ignore_eos': True}
            component_result['component_snapshot'] = snapshot_components(engine, tracking)
        name = nvml.nvmlDeviceGetName(handle)
        return {
            **component_result,
            **summarize_samples(sampler.samples, baseline['used_bytes']),
            'measurement': 'sampled NVML total device used memory',
            'gpu_uuid': uuid, 'gpu_name': name.decode() if isinstance(name, bytes) else name,
            'gpu_total_gib': baseline['total_bytes'] / GIB,
            'requested_sampling_interval_ms': SAMPLE_INTERVAL_SECONDS * 1000,
            'workload': {'input_tokens': INPUT_TOKENS, 'output_tokens': OUTPUT_TOKENS,
                         'batch_size': 1, 'warmup_requests': WARMUP_REQUESTS,
                         'measured_requests': MEASURED_REQUESTS, 'generated_token_counts': generated_counts,
                         'ignore_eos': True, 'temperature': 0.0},
            'engine_settings': ENGINE_SETTINGS,
            'allocator_peaks': ({'status': 'ok', 'peak_bytes': tracking['peak_allocator_bytes'],
                                 'reserved_peak_bytes': tracking['peak_allocator_reserved_bytes'],
                                 'scope': 'worker allocator peak during separate instrumented inference'} if tracking else
                                {'status': 'unsupported', 'reason': 'Component instrumentation was not requested.'}),
            'limitations': [
                'Sampling can miss allocations shorter than the observed sampling gaps.',
                'Device-wide memory includes driver/runtime overhead and the preallocated vLLM KV pool.',
                'Baseline subtraction is an observed increase, not exact model-process attribution.',
                'CUDA device discovery occurs before baseline measurement; its initialization is excluded.',
                'This fixed workload does not establish matched-quality or matched-cache-budget comparisons.',
            ],
            'samples': sampler.samples,
        }
    except nvml.NVMLError as error:
        return {'status': 'unsupported', 'reason': f'Required NVML query unavailable: {error}'}
    finally:
        nvml.nvmlShutdown()
