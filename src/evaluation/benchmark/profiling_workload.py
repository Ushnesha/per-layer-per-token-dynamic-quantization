"""Internal workload launched under Nsight Systems; mark actual vLLM worker execution."""

from .worker_extension import WORKER_EXTENSION

import json
import os
from pathlib import Path
import sys
from types import MethodType
from contextlib import contextmanager
import tempfile

from .perplexity import ENGINE_SETTINGS

INPUT_TOKENS = 512
OUTPUT_TOKENS = 8
STEP_LIMITS = {'prefill': 1, 'decode': 3}
PROFILING_SETTINGS = {
    **ENGINE_SETTINGS, 'enforce_eager': True, 'async_scheduling': False,
    'enable_chunked_prefill': False, 'max_num_seqs': 1,
}


@contextmanager
def profile_artifacts(label):
    """Keep profiler evidence after Slurm completion or failure."""
    job = os.environ.get('SLURM_JOB_ID')
    if job:
        path = Path(__file__).resolve().parents[3] / 'results' / 'benchmark' / 'profiles' / job / label
        path.mkdir(parents=True, exist_ok=True)
        yield str(path)
    else:
        with tempfile.TemporaryDirectory(prefix=f'benchmark-{label}-') as path:
            yield path


def install_ranges(worker):
    import torch
    runner = worker.model_runner
    if hasattr(worker, '_aditya_profile_ranges'):
        raise RuntimeError('Profiling ranges are already installed.')
    original = runner.execute_model
    state = {'original': original,
             'prefill': {'steps': 0, 'processed_tokens': 0},
             'decode': {'steps': 0, 'processed_tokens': 0}}
    worker._aditya_profile_ranges = state
    def execute(self, scheduler_output, *args, **kwargs):
        scheduled = list(scheduler_output.num_scheduled_tokens.values())
        if not scheduled:
            return original(scheduler_output, *args, **kwargs)
        if len(scheduled) != 1 or scheduled[0] <= 0:
            raise ValueError('Profiler requires one active request with positive scheduled tokens.')
        phase = 'prefill' if scheduled[0] > 1 else 'decode'
        marked = state[phase]['steps'] < STEP_LIMITS[phase]
        if not marked:
            return original(scheduler_output, *args, **kwargs)
        torch.cuda.synchronize(worker.device)
        torch.cuda.nvtx.range_push(f'aditya_{phase}')
        try:
            result = original(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize(worker.device)
            state[phase]['steps'] += 1
            state[phase]['processed_tokens'] += scheduled[0]
            return result
        finally:
            torch.cuda.nvtx.range_pop()
    runner.execute_model = MethodType(execute, runner)
    try:
        torch.cuda.profiler.start()
    except Exception:
        runner.execute_model = original
        del worker._aditya_profile_ranges
        raise
    return {'status': 'ok'}


def remove_ranges(worker):
    import torch
    state = worker._aditya_profile_ranges
    try:
        torch.cuda.synchronize(worker.device)
        torch.cuda.profiler.stop()
    finally:
        worker.model_runner.execute_model = state['original']
        del worker._aditya_profile_ranges
    return {phase: state[phase] for phase in STEP_LIMITS}


def run_profile_workload(model_path: str) -> dict:
    os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    body = tokenizer('Describe the history and natural features of the city. ', add_special_tokens=False)['input_ids']
    if not body or tokenizer.bos_token_id is None:
        raise ValueError('Cannot construct the kernel-profiling workload.')
    tokens = [tokenizer.bos_token_id] + (body * (INPUT_TOKENS // len(body) + 1))[:INPUT_TOKENS - 1]
    engine = LLM(model=model_path, worker_extension_cls=WORKER_EXTENSION, **PROFILING_SETTINGS)
    print('Profiler engine loaded; starting warmup', flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True)
    prompt = [TokensPrompt(prompt_token_ids=tokens)]
    engine.generate(prompt, params, use_tqdm=False)  # warm-up is outside profiler ranges
    print('Warmup complete; installing worker NVTX ranges', flush=True)
    engine.collective_rpc('aditya_install_ranges', timeout=60)
    try:
        outputs = engine.generate(prompt, params, use_tqdm=False)
    finally:
        stats = engine.collective_rpc('aditya_remove_ranges', timeout=60)
    if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != tokens or len(outputs[0].outputs) != 1:
        raise ValueError('Profiling workload returned altered or incomplete prompts.')
    if len(outputs[0].outputs[0].token_ids) != OUTPUT_TOKENS:
        raise ValueError('Profiling workload generated an unexpected output length.')
    for worker in stats:
        if worker['prefill']['steps'] != 1 or worker['prefill']['processed_tokens'] != INPUT_TOKENS:
            raise ValueError('Expected one full prefill step.')
        if worker['decode']['steps'] != 3 or worker['decode']['processed_tokens'] != 3:
            raise ValueError('Expected three single-token decode steps.')
    return {'input_tokens': INPUT_TOKENS, 'output_tokens': OUTPUT_TOKENS,
            'batch_size': 1, 'warmup_requests': 1, 'measured_requests': 1,
            'engine_settings': PROFILING_SETTINGS, 'worker_stats': stats}


def main():
    model_path, destination = sys.argv[1:]
    result = run_profile_workload(model_path)
    with Path(destination).open('w') as handle:
        json.dump(result, handle, allow_nan=False)


if __name__ == '__main__':
    from src.evaluation.benchmark.profiling_workload import main as run_named_main
    run_named_main()
