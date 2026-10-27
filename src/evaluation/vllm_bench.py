"""
vLLM benchmark worker: WikiText-2 perplexity and latency for one model.

Every method (FP16, AWQ, GPTQ, SmoothQuant) runs through the same vLLM engine
settings, so each number reflects the kernel that would actually serve it
(cuBLAS FP16, awq_marlin, gptq_marlin, INT8 W8A8 CUTLASS).

Run as a subprocess, one model per process, so GPU memory is fully released:
    python -m src.evaluation.vllm_bench --model PATH --out bench.json [--skip-ppl] [--skip-latency]

vLLM logs "Model loading took X GiB" to stderr; the evaluator parses that for
the measured weight memory.
"""

import argparse
import json
import math
import os
import statistics
import time

# FlashInfer's sampler is JIT-compiled with the host GCC (8.5 here, which fails);
# use vLLM's PyTorch sampler for every method. Must be set before importing vllm.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np
from transformers import AutoTokenizer

from src.config.settings import (
    EvalConfig, SEED, VLLM_DTYPE, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION,
)
from src.utils.perplexity import wikitext_windows


def build_llm(model_path: str):
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=VLLM_DTYPE,
        max_model_len=VLLM_MAX_MODEL_LEN,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        seed=SEED,
        enable_prefix_caching=False,  # every latency iteration must redo prefill
    )


def vllm_ppl(llm, tokenizer, seq_len: int) -> dict:
    """PPL from prompt logprobs over the shared non-overlapping WikiText-2 windows."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    windows = wikitext_windows(tokenizer, seq_len)
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in windows], params, use_tqdm=True)

    total_nll = 0.0
    total_tokens = 0
    for window, out in zip(windows, outputs):
        # prompt_logprobs[0] is None (no prediction for the first token)
        for pos in range(1, len(window)):
            total_nll -= out.prompt_logprobs[pos][window[pos]].logprob
            total_tokens += 1

    return {
        "wikitext": round(math.exp(total_nll / total_tokens), 4),
        "seq_len": seq_len,
        "num_windows": len(windows),
        "num_tokens": total_tokens,
    }


def vllm_latency(llm, tokenizer, cfg: EvalConfig) -> dict:
    """Batch-1 latency: TTFT = median time for 1 output token; TPOT = (E2E - TTFT) / (N - 1)."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    rng = np.random.default_rng(SEED)
    prompt_ids = rng.integers(1000, tokenizer.vocab_size - 1000, size=cfg.prompt_length).tolist()
    prompt = [TokensPrompt(prompt_token_ids=prompt_ids)]

    def timed(max_tokens: int) -> float:
        params = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
        start = time.perf_counter()
        llm.generate(prompt, params, use_tqdm=False)
        return time.perf_counter() - start

    for _ in range(cfg.warmup_iters):
        timed(cfg.output_length)
        timed(1)

    ttft_runs = [timed(1) for _ in range(cfg.measure_iters)]
    e2e_runs = [timed(cfg.output_length) for _ in range(cfg.measure_iters)]
    ttft = statistics.median(ttft_runs)
    e2e = statistics.median(e2e_runs)

    return {
        "batch_size": 1,
        "input_len": cfg.prompt_length,
        "output_len": cfg.output_length,
        "iters": cfg.measure_iters,
        "ttft_ms": round(ttft * 1000, 3),
        "tpot_ms": round((e2e - ttft) / (cfg.output_length - 1) * 1000, 3),
        "e2e_ms": round(e2e * 1000, 3),
        "e2e_ms_stdev": round(statistics.stdev(e2e_runs) * 1000, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM PPL + latency benchmark for one model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    cfg = EvalConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = build_llm(args.model)

    import vllm
    model_config = llm.llm_engine.model_config
    results = {"engine": {
        "vllm_version": vllm.__version__,
        "dtype": str(model_config.dtype),
        "quantization": model_config.quantization,
        "max_model_len": VLLM_MAX_MODEL_LEN,
        "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
    }}

    if not args.skip_ppl:
        results["perplexity"] = vllm_ppl(llm, tokenizer, cfg.ppl_seq_len)
    if not args.skip_latency:
        results["latency"] = vllm_latency(llm, tokenizer, cfg)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
