"""Independent WikiText-2 test perplexity using vLLM prompt log probabilities.

This measures prefill prediction quality, not quantized KV-cache decoding.
"""

import math
from typing import Any, Sequence

DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
CONTEXT_LENGTH = 2048
ENGINE_SETTINGS = {
    "dtype": "float16",
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "enable_prefix_caching": False,
    "seed": 0,
}


def make_windows(token_ids: Sequence[int], bos_token_id: int,
                 context_length: int = CONTEXT_LENGTH) -> list[list[int]]:
    """Prepend BOS to disjoint full blocks; exclude the trailing partial block."""
    if bos_token_id is None:
        raise ValueError("The tokenizer must define bos_token_id for this protocol.")
    if context_length < 2:
        raise ValueError("Context length must be at least two tokens.")
    body_length = context_length - 1
    count = len(token_ids) // body_length
    if count == 0:
        raise ValueError("Dataset does not contain a full evaluation window.")
    return [[bos_token_id] + list(token_ids[i * body_length:(i + 1) * body_length])
            for i in range(count)]


def score_outputs(windows: Sequence[Sequence[int]], outputs: Sequence[Any]) -> dict:
    """Aggregate actual-token logprobs, excluding BOS and generated tokens."""
    if not windows or len(windows) != len(outputs):
        raise ValueError("Expected one output for every nonempty evaluation window.")
    total_nll = 0.0
    count = 0
    for window, output in zip(windows, outputs):
        if list(output.prompt_token_ids) != list(window):
            raise ValueError("Returned prompt tokens differ from the evaluation window.")
        logprobs = output.prompt_logprobs
        if logprobs is None or len(logprobs) != len(window):
            raise ValueError("Missing or incomplete prompt log probabilities.")
        for position, token in enumerate(window[1:], start=1):
            scores = logprobs[position]
            if scores is None or token not in scores:
                raise ValueError(f"Missing actual-token log probability at position {position}.")
            logprob = float(scores[token].logprob)
            if not math.isfinite(logprob) or logprob > 0:
                raise ValueError(f"Invalid log probability at position {position}: {logprob}")
            total_nll -= logprob
            count += 1
    if count == 0:
        raise ValueError("No tokens were scored.")
    mean_nll = total_nll / count
    value = math.exp(mean_nll)
    if not math.isfinite(value):
        raise ValueError("Perplexity is not finite.")
    return {"value": value, "total_nll": total_nll,
            "mean_nll": mean_nll, "scored_tokens": count}


def evaluate_perplexity(model_path: str) -> dict:
    # Lazy imports let CLI help and mathematical tests run without CUDA/vLLM.
    import os
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    dataset = load_dataset(DATASET, DATASET_CONFIG, split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False)["input_ids"]
    windows = make_windows(token_ids, tokenizer.bos_token_id)
    available_windows = len(windows)
    engine = LLM(model=model_path, **ENGINE_SETTINGS)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    # Bound host memory used by prompt-logprob outputs, independent of dataset size.
    total_nll = 0.0
    scored_tokens = 0
    for start in range(0, len(windows), 8):
        batch = windows[start:start + 8]
        outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=window) for window in batch],
            params, use_tqdm=False,
        )
        score = score_outputs(batch, outputs)
        total_nll += score["total_nll"]
        scored_tokens += score["scored_tokens"]

    return {
        "status": "ok",
        "value": math.exp(total_nll / scored_tokens),
        "mean_nll": total_nll / scored_tokens,
        "total_nll": total_nll,
        "scored_tokens": scored_tokens,
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "split": "test",
        "context_length": CONTEXT_LENGTH,
        "num_windows": len(windows),
        "available_windows": available_windows,
        "omitted_windows": available_windows - len(windows),
        "dropped_tokens": len(token_ids) - scored_tokens,
        "bos_token_id": tokenizer.bos_token_id,
        "protocol": "disjoint BOS + 2047 text tokens; trailing partial block dropped",
        "scoring_path": "prefill_prompt_logprobs",
    }
