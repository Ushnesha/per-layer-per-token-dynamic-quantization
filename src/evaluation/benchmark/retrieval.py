"""Synthetic passkey retrieval through vLLM generation (not the RULER suite).

Baseline cache evaluation only: custom/static quantized KV backends and matched
cache-budget comparisons must be integrated before drawing RQ2 conclusions.
"""

import os
import random
import re

from .perplexity import ENGINE_SETTINGS
from .instruction_prompt import chat_parts, prompt_protocol, stop_token_ids

CONTEXT_LENGTHS = (1024, 2048, 4096, 7168)
TARGET_POSITIONS = (0.1, 0.5, 0.9)
EXAMPLES_PER_CELL = 100
OUTPUT_TOKENS = 32
SEED = 0
FILLER = (
    "The archive contains ordinary notes about the weather and daily activities. ",
    "People walked through the garden and discussed plans for the afternoon. ",
    "The library shelves held books about history, nature, and distant places. ",
    "A quiet morning passed as the village prepared for another working day. ",
)


def exact_match(response: str, expected: str) -> bool:
    """Require only the correct eight-digit key, allowing surrounding whitespace."""
    return re.fullmatch(r"[0-9]{8}", response.strip()) is not None and response.strip() == expected


def build_example(tokenizer, context_length: int, position: float, seed: int) -> dict:
    """Construct exactly context_length input tokens without truncating the fact."""
    if not 0 <= position <= 1:
        raise ValueError("Target position must be between zero and one.")
    if tokenizer.bos_token_id is None:
        raise ValueError("The tokenizer must define a BOS token.")
    rng = random.Random(seed)
    key = str(rng.randrange(10_000_000, 100_000_000))
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    prefix, suffix = chat_parts(tokenizer,
        "Read the archive below. Remember its secret passkey and answer the final question.\n\n",
        "\n\nWhat is the secret passkey? Reply with only its eight digits.")
    fact = encode(f"\nThe secret passkey is {key}. Remember this exact passkey.\n")
    filler_length = context_length - len(prefix) - len(fact) - len(suffix)
    if filler_length < 0:
        raise ValueError("Context is too short to fit instructions, fact, and question.")
    filler = []
    while len(filler) < filler_length:
        block = encode(rng.choice(FILLER))
        if not block:
            raise ValueError("Tokenizer produced empty distractor tokens.")
        filler.extend(block)
    filler = filler[:filler_length]
    insertion = round(position * filler_length)
    tokens = prefix + filler[:insertion] + fact + filler[insertion:] + suffix
    assert len(tokens) == context_length
    return {
        "prompt_token_ids": tokens,
        "expected": key,
        "seed": seed,
        "target_position": position,
        "target_token_index": len(prefix) + insertion,
        "actual_target_fraction": (len(prefix) + insertion) / context_length,
    }


def summarize_trials(trials: list[dict]) -> dict:
    if not trials:
        raise ValueError("Retrieval evaluation contains no trials.")
    cells = {}
    for trial in trials:
        if not isinstance(trial["correct"], bool):
            raise ValueError("Each trial must contain a boolean correctness result.")
        group = (trial["context_length"], trial["target_position"])
        cells.setdefault(group, []).append(trial["correct"])
    breakdown = []
    for (length, position), outcomes in sorted(cells.items()):
        breakdown.append({"context_length": length, "target_position": position,
                          "correct": sum(outcomes), "total": len(outcomes),
                          "accuracy": 100 * sum(outcomes) / len(outcomes)})
    correct = sum(trial["correct"] for trial in trials)
    return {"value": 100 * correct / len(trials), "unit": "%",
            "correct": correct, "total": len(trials), "by_context_and_position": breakdown}


def evaluate_retrieval(model_path: str) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Let vLLM resolve the model's supported context rather than inheriting the
    # 4096-token engine cap used by the short-context PPL/accuracy benchmarks.
    settings = {**ENGINE_SETTINGS, "max_model_len": None}
    engine = LLM(model=model_path, **settings)
    model_limit = int(engine.llm_engine.model_config.max_model_len)
    lengths = sorted({min(length, model_limit - OUTPUT_TOKENS) for length in CONTEXT_LENGTHS})
    if not lengths or lengths[0] <= 0:
        raise ValueError("Model context limit cannot accommodate the output reservation.")
    params = SamplingParams(temperature=0.0, max_tokens=OUTPUT_TOKENS,
                            stop_token_ids=stop_token_ids(tokenizer), stop=['\n'])
    trials = []
    for length_index, length in enumerate(lengths):
        for position_index, position in enumerate(TARGET_POSITIONS):
            for index in range(EXAMPLES_PER_CELL):
                seed = SEED + length_index * 100_000 + position_index * 1000 + index
                example = build_example(tokenizer, length, position, seed)
                outputs = engine.generate(
                    [TokensPrompt(prompt_token_ids=example["prompt_token_ids"])],
                    params, use_tqdm=False,
                )
                if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != example["prompt_token_ids"]:
                    raise ValueError("Retrieval prompt was truncated or returned out of order.")
                if len(outputs[0].outputs) != 1:
                    raise ValueError("Expected one generated answer per retrieval query.")
                answer = outputs[0].outputs[0]
                trials.append({
                    **{key: value for key, value in example.items() if key != "prompt_token_ids"},
                    "context_length": length, "response": answer.text,
                    "generated_tokens": len(answer.token_ids),
                    "finish_reason": answer.finish_reason,
                    "correct": exact_match(answer.text, example["expected"]),
                })
    return {
        "status": "ok", **summarize_trials(trials),
        "benchmark": "synthetic_passkey_v2_chat", "seed": SEED,
        "requested_context_lengths": list(CONTEXT_LENGTHS),
        "context_lengths": lengths, "model_context_limit": model_limit,
        "target_positions": list(TARGET_POSITIONS),
        "position_definition": "fraction of distractor tokens before the fact",
        "examples_per_cell": EXAMPLES_PER_CELL, "output_tokens": OUTPUT_TOKENS,
        "batch_size": 1, **prompt_protocol(tokenizer), "stop_strings": ['\n'],
        "scoring": "exact eight-digit answer after trimming surrounding whitespace",
        "engine_settings": {**settings, "max_model_len": model_limit},
        "cache_evaluation": {
            "mode": "vllm_default_cache_baseline",
            "quantized_cache_verified": False,
            "matched_cache_budget_comparison": "blocked: cache backends and memory controls not integrated",
        },
        "trials": trials,
    }
