"""Fresh-process search for maximum memory-feasible prompt + output length."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .perplexity import ENGINE_SETTINGS
from .serving import build_prompt, stop_server
from .serving_memory import serving_memory

OUTPUT_TOKENS = 128
GRID_STEP = 128
CONFIRMATIONS = 3
PROBE_TIMEOUT = 600
SEARCH_TIMEOUT = 5400
# Reserve five percentage points for driver/graph/startup variation under the
# unchanged 85% external device-growth cap. Use the same policy for every model.
ENGINE_HEADROOM_FRACTION = 0.05


def candidate_lengths(model_limit: int) -> list[int]:
    if type(model_limit) is not int or model_limit <= OUTPUT_TOKENS:
        raise ValueError('Model context must fit at least one input token plus the output reservation.')
    return sorted({OUTPUT_TOKENS + 1, model_limit,
                   *range(OUTPUT_TOKENS + GRID_STEP, model_limit + 1, GRID_STEP)})


def memory_failure(text: str):
    lower = text.lower()
    if any(marker in lower for marker in ('cuda out of memory', 'cuda error: out of memory', 'torch.outofmemoryerror')):
        return 'cuda_out_of_memory'
    if ('no available memory for the cache blocks' in lower
            or ('larger than the available kv cache memory' in lower)
            or ('maximum number of tokens that can be stored in kv cache' in lower)):
        return 'insufficient_kv_capacity'
    return None


def resolve_model_limit(model_path: str) -> dict:
    # Configuration-only resolution applies vLLM's model/rope rules without
    # allocating a full-context engine that might itself fail to fit.
    from vllm.config import ModelConfig
    config = ModelConfig(model=model_path, tokenizer=model_path,
                         dtype=ENGINE_SETTINGS['dtype'], seed=ENGINE_SETTINGS['seed'], max_model_len=None)
    return {'status': 'ok', 'model_context_limit': int(config.max_model_len)}


def probe_context(model_path: str, total_tokens: int) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    if total_tokens <= OUTPUT_TOKENS:
        raise ValueError('Context must reserve at least one input token.')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokens = build_prompt(tokenizer, total_tokens - OUTPUT_TOKENS)
    settings = {**ENGINE_SETTINGS, 'max_model_len': total_tokens, 'max_num_seqs': 1,
                'gpu_memory_utilization': ENGINE_SETTINGS['gpu_memory_utilization'] - ENGINE_HEADROOM_FRACTION,
                'cpu_offload_gb': 0, 'offload_group_size': 0, 'kv_offloading_size': None}
    with serving_memory() as monitor:
        if monitor.unavailable is not None:
            return {'verdict': 'inconclusive', 'reason': 'Memory-budget verification unavailable.',
                    'terminal': True, 'memory': monitor.unavailable}
        identity = {'gpu_uuid': monitor.uuid, 'memory_budget_bytes': monitor.budget_bytes}
        capture = None
        try:
            with monitor.capture() as capture:
                capture.set_phase('startup')
                engine = LLM(model=model_path, **settings)
                resolved = int(engine.llm_engine.model_config.max_model_len)
                if resolved != total_tokens:
                    raise ValueError('Backend changed the requested context limit.')
                capture.set_phase('inference')
                outputs = engine.generate([TokensPrompt(prompt_token_ids=tokens)],
                    SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS, ignore_eos=True), use_tqdm=False)
                if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != tokens or len(outputs[0].outputs) != 1:
                    raise ValueError('Context probe returned incomplete or altered prompts.')
                generated = len(outputs[0].outputs[0].token_ids)
                if generated != OUTPUT_TOKENS:
                    raise ValueError('Context probe did not complete the reserved output length.')
            measured = capture.result
            if measured.get('status') != 'ok':
                verdict, reason = 'inconclusive', 'Memory samples are unavailable or contaminated.'
            elif measured['within_budget']:
                verdict, reason = 'pass', None
            else:
                verdict, reason = 'memory_limit', 'Measured device growth exceeded the fixed budget.'
            return {'verdict': verdict, 'reason': reason, **identity, 'memory': measured,
                    'input_tokens': len(tokens), 'output_tokens': generated,
                    'total_tokens': total_tokens, 'engine_settings': settings}
        except Exception as error:
            category = memory_failure(f'{type(error).__name__}: {error}')
            return {'verdict': 'memory_limit' if category else 'inconclusive',
                    'reason': category or str(error), 'error_type': type(error).__name__, **identity,
                    'memory': capture.result if capture is not None else None,
                    'total_tokens': total_tokens, 'engine_settings': settings}


def run_action(action: str, model_path: str, total_tokens=None, timeout=PROBE_TIMEOUT) -> dict:
    with tempfile.TemporaryDirectory(prefix='benchmark-context-') as temporary:
        destination = Path(temporary) / 'result.json'
        command = [sys.executable, '-m', 'src.evaluation.benchmark.max_context', action,
                   model_path, str(destination)]
        if total_tokens is not None:
            command.append(str(total_tokens))
        environment = os.environ.copy()
        environment.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')
        environment.pop('VLLM_ALLOW_LONG_MAX_MODEL_LEN', None)
        with tempfile.TemporaryFile(mode='w+') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=environment,
                                       cwd=Path(__file__).resolve().parents[3])
            timed_out = False
            try:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
            finally:
                stop_server(process)  # Dedicated process group, including vLLM workers.
            log.seek(0)
            diagnostics = log.read()[-8000:]
        if timed_out:
            return {'verdict': 'inconclusive', 'reason': 'Probe timeout; not classified as a memory limit.', 'diagnostics': diagnostics}
        if not destination.is_file():
            return {'verdict': 'inconclusive', 'reason': f'Probe exited without results ({process.returncode}).',
                    'diagnostics': diagnostics}
        result = json.loads(destination.read_text())
        # vLLM may surface only a generic worker-init exception in Python; a
        # memory classification still requires specific GPU/cache evidence.
        category = memory_failure(diagnostics)
        if action == 'probe' and result.get('verdict') == 'inconclusive' and category and result.get('gpu_uuid'):
            result.update(verdict='memory_limit', reason=category)
        if result.get('verdict') not in (None, 'pass'):
            result['diagnostics'] = diagnostics
        return result


def search_context(model_limit: int, probe, deadline=None) -> dict:
    candidates = candidate_lengths(model_limit)
    trials, found = [], None
    identity = None
    budget_mismatch = False
    deadline_reached = False
    terminal = False
    for total in reversed(candidates):
        repetitions = []
        for repetition in range(CONFIRMATIONS):
            if deadline is not None and time.monotonic() >= deadline:
                deadline_reached = True
                break
            result = probe(total)
            repetitions.append({'repetition': repetition, **result})
            terminal = result.get('terminal', False)
            if result.get('gpu_uuid') is not None and result.get('memory_budget_bytes') is not None:
                current_identity = (result['gpu_uuid'], result['memory_budget_bytes'])
                if identity is None:
                    identity = current_identity
                elif identity != current_identity:
                    budget_mismatch = True
                    break
            elif result.get('verdict') == 'pass':
                repetitions[-1].update(verdict='inconclusive', reason='Passing probe lacks GPU/budget identity.')
            if repetitions[-1].get('verdict') != 'pass':
                break
        verified = len(repetitions) == CONFIRMATIONS and all(row.get('verdict') == 'pass' for row in repetitions)
        if budget_mismatch:
            verdict = 'inconclusive'
        elif verified:
            verdict = 'pass'
        elif any(row.get('verdict') == 'memory_limit' for row in repetitions):
            verdict = 'unstable' if any(row.get('verdict') == 'pass' for row in repetitions) else 'memory_limit'
        else:
            verdict = 'inconclusive'
        trials.append({'total_tokens': total, 'input_tokens': total - OUTPUT_TOKENS,
                       'verdict': verdict, 'probes': repetitions})
        if budget_mismatch or deadline_reached or terminal:
            break
        if verified:
            found = total
            break
    higher = [row for row in trials if found is not None and row['total_tokens'] > found]
    if budget_mismatch:
        interpretation = 'inconclusive: GPU or memory budget changed during the search'
        found = None
    elif found == model_limit:
        interpretation = 'model_supported_limit_reached; no lower memory limit was observed'
    elif found is not None and all(row['verdict'] == 'memory_limit' for row in higher):
        interpretation = 'memory_limited_on_tested_grid'
    elif found is not None:
        interpretation = 'verified_lower_bound; higher lengths were unstable or inconclusive'
    elif deadline_reached:
        interpretation = 'search_deadline_reached_without_a_verified_length'
    elif trials and all(row['verdict'] == 'memory_limit' for row in trials):
        interpretation = 'no_feasible_length_on_tested_grid'
    else:
        interpretation = 'inconclusive: no length passed every confirmation'
    return {
        'status': 'ok' if found is not None or interpretation == 'no_feasible_length_on_tested_grid' else 'blocked',
        'max_total_tokens': found, 'max_input_tokens': found - OUTPUT_TOKENS if found is not None else None,
        'output_tokens': OUTPUT_TOKENS, 'unit': 'tokens', 'interpretation': interpretation,
        'model_context_limit': model_limit, 'grid_step_tokens': GRID_STEP,
        'minimum_total_tokens': candidates[0], 'required_confirmations': CONFIRMATIONS,
        'budget_identity': {'gpu_uuid': identity[0], 'memory_budget_bytes': identity[1]} if identity else None,
        'trials': trials,
    }


def evaluate_max_context(model_path: str) -> dict:
    deadline = time.monotonic() + SEARCH_TIMEOUT
    resolved = run_action('resolve', model_path)
    if resolved.get('status') != 'ok':
        return {'status': 'blocked', 'reason': 'Could not resolve the model-supported context length.', 'resolution': resolved}
    if resolved['model_context_limit'] <= OUTPUT_TOKENS:
        return {'status': 'unsupported', 'reason': 'Model context cannot fit the fixed output reservation plus an input token.',
                'model_context_limit': resolved['model_context_limit'], 'output_tokens': OUTPUT_TOKENS}
    def probe(total):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {'verdict': 'inconclusive', 'reason': 'Search deadline reached.'}
        return run_action('probe', model_path, total, timeout=min(PROBE_TIMEOUT, remaining))
    search_limit = resolved['model_context_limit']
    result = search_context(search_limit, probe, deadline)
    return {
        **result, 'batch_size': 1,
        'search': 'descending grid; each probe runs in a fresh process',
        'probe_timeout_seconds': PROBE_TIMEOUT, 'search_timeout_seconds': SEARCH_TIMEOUT,
        'memory_budget_fraction': ENGINE_SETTINGS['gpu_memory_utilization'],
        'engine_pool_fraction': ENGINE_SETTINGS['gpu_memory_utilization'] - ENGINE_HEADROOM_FRACTION,
        'engine_headroom_fraction': ENGINE_HEADROOM_FRACTION,
        'memory_budget_scope': 'sampled device growth above an exclusive pre-engine baseline; startup and generation included',
        'limitations': [
            'Measures memory feasibility and successful generation, not quality-effective context length.',
            ('Grid points are 128-token total-length increments plus the exact model limit and one-token-input minimum.'),
            f'A passing length requires {CONFIRMATIONS} fresh-process success(es) at the same GPU/budget identity.',
            'Timeouts, unsupported kernels, and generic worker failures are inconclusive rather than memory failures.',
            'Device-memory sampling can miss brief peaks; CPU weight/KV offloading and prefix caching are disabled.',
            'Static/adaptive KV comparisons require their real cache backends and equivalent budget settings.',
        ],
    }


if __name__ == '__main__':
    action, model_path, destination, *extra = sys.argv[1:]
    try:
        result = resolve_model_limit(model_path) if action == 'resolve' else probe_context(model_path, int(extra[0]))
    except Exception as error:
        result = {'status': 'failed', 'verdict': 'inconclusive', 'error_type': type(error).__name__, 'reason': str(error)}
    with Path(destination).open('w') as handle:
        json.dump(result, handle, allow_nan=False)
