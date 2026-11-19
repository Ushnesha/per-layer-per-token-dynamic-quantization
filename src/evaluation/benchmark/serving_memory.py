"""Optional NVML monitoring of the actual serving trials, before/after load."""

from contextlib import contextmanager
import os

from .peak_memory import MemorySampler, SAMPLE_INTERVAL_SECONDS
from .perplexity import ENGINE_SETTINGS


class TrialMemory:
    def __init__(self, result, sampler=None):
        self.result = result
        self.sampler = sampler
    def set_phase(self, phase):
        if self.sampler is not None:
            try:
                self.sampler.set_phase(phase)
            except Exception as error:
                self.result = {'status': 'failed', 'reason': str(error)}
                self.sampler.errors.append(str(error))
                self.sampler.finished.set()


def memory_result(samples, baseline, budget_bytes, gpu_uuid):
    if not samples:
        return {'status': 'unsupported', 'reason': 'No serving memory samples.'}
    if (not 0 <= baseline['used_bytes'] <= baseline['total_bytes'] or baseline['total_bytes'] <= 0
            or type(budget_bytes) is not int or budget_bytes <= 0
            or any(not 0 <= row['used_bytes'] <= row['total_bytes']
                   or row['total_bytes'] != baseline['total_bytes'] for row in samples)):
        raise ValueError('Invalid serving memory samples or budget.')
    peak = max(row['used_bytes'] for row in samples)
    foreign = sorted({pid for row in samples for pid in row['foreign_pids']})
    growth = max(0, peak - baseline['used_bytes'])
    gaps = [b['seconds'] - a['seconds'] for a, b in zip(samples, samples[1:])]
    return {
        'status': 'blocked' if foreign else 'ok', 'gpu_uuid': gpu_uuid,
        'baseline_device_bytes': baseline['used_bytes'], 'peak_device_bytes': peak,
        'observed_device_growth_bytes': growth, 'memory_budget_bytes': budget_bytes,
        'within_budget': growth <= budget_bytes and peak <= baseline['total_bytes'] if not foreign else None,
        'budget_scope': 'device memory increase above exclusive pre-server baseline',
        'gpu_total_bytes': baseline['total_bytes'], 'foreign_gpu_pids': foreign,
        'requested_sample_interval_ms': SAMPLE_INTERVAL_SECONDS * 1000,
        'maximum_sample_gap_ms': max(gaps, default=0) * 1000, 'num_samples': len(samples),
    }


class ServingMemory:
    def __init__(self):
        self.nvml = None
        self.unavailable = {'status': 'unsupported', 'reason': 'Serving GPU memory monitoring unavailable.'}
        try:
            import pynvml as nvml
            import psutil
            import torch
            if not torch.cuda.is_available():
                self.unavailable['reason'] = 'CUDA GPU required for serving memory monitoring.'
                return
            uuid = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
            if uuid is None:
                self.unavailable['reason'] = 'CUDA UUID unavailable; refusing to guess NVML device.'
                return
            self.uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
            if self.uuid.startswith('MIG-'):
                self.unavailable['reason'] = 'MIG accounting is not supported.'
                return
            nvml.nvmlInit()
            self.nvml = nvml
            self.handle = nvml.nvmlDeviceGetHandleByUUID(self.uuid)
            self.parent = psutil.Process(os.getpid())
            self.baseline = self.read()
            self.budget_bytes = int(self.baseline['total_bytes'] * ENGINE_SETTINGS['gpu_memory_utilization'])
            if self.baseline['foreign_pids']:
                self.unavailable = {'status': 'blocked', 'reason': 'Serving memory requires an exclusive GPU.',
                                    'foreign_gpu_pids': self.baseline['foreign_pids']}
                return
            self.unavailable = None
        except Exception as error:
            self.unavailable = {'status': 'unsupported', 'reason': str(error)}

    def read(self):
        info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        own_pids = {self.parent.pid, *(child.pid for child in self.parent.children(recursive=True))}
        processes = [*self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle),
                     *self.nvml.nvmlDeviceGetGraphicsRunningProcesses(self.handle)]
        return {'used_bytes': int(info.used), 'total_bytes': int(info.total),
                'foreign_pids': sorted({process.pid for process in processes if process.pid not in own_pids})}

    @contextmanager
    def capture(self):
        if self.unavailable is not None:
            yield TrialMemory(dict(self.unavailable))
            return
        sampler = MemorySampler(self.read)
        try:
            sampler.start()
            sampler.set_phase('warmup')
        except Exception as error:
            if sampler.thread.is_alive():
                sampler.finished.set()
                sampler.thread.join(timeout=5)
            yield TrialMemory({'status': 'failed', 'reason': str(error)})
            return
        result = TrialMemory({}, sampler)
        try:
            yield result
        finally:
            try:
                sampler.stop()
                result.result = memory_result(sampler.samples, self.baseline, self.budget_bytes, self.uuid)
            except Exception as error:
                result.result = {'status': 'failed', 'reason': str(error)}

    def close(self):
        if self.nvml is not None:
            self.nvml.nvmlShutdown()


@contextmanager
def serving_memory():
    monitor = ServingMemory()
    try:
        yield monitor
    finally:
        monitor.close()
