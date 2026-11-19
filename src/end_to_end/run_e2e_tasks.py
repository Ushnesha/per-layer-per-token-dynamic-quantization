"""End-to-end long-context tests, using Aditya's tasks and scoring.

The same weights x KV-cache grid as run_e2e.py, on the two long-context tests from
src/evaluation/benchmark where the KV cache matters:

    - Passkey retrieval (retrieval.py): Aditya's synthetic examples with 2,048- and 7,168-token
      prompts, the passkey at 10/50/90% depth, 10 examples per cell (60 in total), with the same
      seeds as his full benchmark. Scored two ways: "found" (the passkey appears in the answer)
      and his exact_match (the answer is only the eight digits).
    - LongBench HotpotQA (long_context.py): the first 50 examples, trimmed to 7,168 tokens with
      his prepare_prompt and scored with his score_prediction (token F1, 0-100).

Why not run Aditya's evaluate.py directly: it runs everything inside vLLM, which uses its own KV
cache, so the per-token KV cache (a Hugging Face cache) would never be used. Here his examples,
prompts, stop tokens and scorers are reused, and generation runs in Hugging Face with
RealQuantKVCache (prototype v1).

Also recorded for each combination:
    - how much of the passkey sentence the KV policy kept in full precision
    - average decode time per token. All weights are unpacked to FP16 here, so this only shows
      the cost of the KV cache, not any speedup from quantized weights.
    - the KV memory the cache really holds (stored_bytes) at the end of the 7,168-token prompts

Results are saved after every combination, so a run that stops early keeps what finished.

Run from the repo root:
    python -m src.end_to_end.run_e2e_tasks --smoke   # 2-3 minute check: FP16 weights, 2 + 2 examples
    python -m src.end_to_end.run_e2e_tasks           # full run, roughly 45-60 minutes on an A100
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the LongBench download hangs with xet

import argparse
import gc
import json
import zipfile

import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from src.end_to_end.quantized_weights import load_model  # also puts src/per_token_quant on the path
from src.evaluation.benchmark.instruction_prompt import stop_token_ids
from src.evaluation.benchmark.long_context import DATASET as LONGBENCH
from src.evaluation.benchmark.long_context import TASKS, prepare_prompt, read_task, score_prediction
from src.evaluation.benchmark.retrieval import (
    CONTEXT_LENGTHS, OUTPUT_TOKENS, SEED, TARGET_POSITIONS, build_example, exact_match,
)
from oracle.quant import FULL_PRECISION
from prototype.adaptive_kv import KeyNormLinearScorer, PositionScorer, generate
from prototype.real_kv import RealQuantKVCache

KEY_NORM_CKPT = "src/per_token_quant/l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
OUTPUT_PATH = "results/end_to_end/e2e_long_context.json"

# name: (checkpoint, weight storage in GB from Ushnesha's results/*.json). Same as run_e2e.py.
WEIGHTS = {
    "FP16": ("models/llama-3-8b-instruct", 16.061),
    "GPTQ W8 (static)": ("models/gptq/llama3-8b-instruct-w8a16", 9.25),
    "GPTQ W4 (static)": ("models/gptq/llama3-8b-instruct-w4a16", 5.733),
    "Mixed ref-kl-5.50 (per-layer)": ("models/mixed/llama3-8b-instruct-ref-kl-5.50", 5.499),
}
# name: (scorer factory, fraction of the tokens leaving the window kept in full precision)
KV_SETTINGS = {
    "full precision": (PositionScorer, 1.0),
    "static INT2": (PositionScorer, 0.0),
    "key norm, 10% kept (per-token)": (lambda: KeyNormLinearScorer(KEY_NORM_CKPT), 0.1),
}
LOW_BITS, GROUP_SIZE = 2, None  # INT2 with one group per head (128 values), as in run_e2e.py
WINDOW, NUM_SINKS = 128, 4

PASSKEY_LENGTHS = (2048, 7168)
PASSKEY_PER_CELL = 10
HOTPOTQA_EXAMPLES = 50
HOTPOTQA_BUDGET = 7168


def passkey_examples(tokenizer, per_cell: int = PASSKEY_PER_CELL, positions=TARGET_POSITIONS) -> list[dict]:
    """Aditya's passkey examples, with the seeds his full benchmark uses for the same cells."""
    examples = []
    for length in PASSKEY_LENGTHS:
        length_index = CONTEXT_LENGTHS.index(length)
        for position in positions:
            position_index = TARGET_POSITIONS.index(position)
            for index in range(per_cell):
                seed = SEED + length_index * 100_000 + position_index * 1000 + index
                example = build_example(tokenizer, length, position, seed)
                example["context_length"] = length
                # The same sentence build_example inserts, to find it in the cache's bit map
                fact = f"\nThe secret passkey is {example['expected']}. Remember this exact passkey.\n"
                example["fact_tokens"] = len(tokenizer(fact, add_special_tokens=False)["input_ids"])
                examples.append(example)
    return examples


def hotpotqa_examples(tokenizer, num_examples: int = HOTPOTQA_EXAMPLES) -> list[tuple[list[int], list[str]]]:
    """(prompt token IDs, reference answers) for the first LongBench HotpotQA examples."""
    path = hf_hub_download(repo_id=LONGBENCH, filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(path) as archive:
        rows = read_task(archive, "hotpotqa")[:num_examples]
    return [(prepare_prompt(tokenizer, "hotpotqa", row, HOTPOTQA_BUDGET)["prompt_token_ids"], row["answers"])
            for row in rows]


def run_passkey(model, tokenizer, examples, make_cache) -> dict:
    found, exact, fact_kept, decode_ms, kv_mib = {}, [], [], [], []
    stops = stop_token_ids(tokenizer)
    for example in tqdm(examples, desc="passkey"):
        out = generate(model, tokenizer, example["prompt_token_ids"], make_cache,
                       max_new_tokens=OUTPUT_TOKENS, stop_token_ids=stops, stop_text="\n")
        cell = f"{example['context_length']} @ {example['target_position']:.0%}"
        found.setdefault(cell, []).append(example["expected"] in out["text"])
        exact.append(exact_match(out["text"], example["expected"]))

        cache = out["cache"]
        start = example["target_token_index"]
        bits = cache.bits[start:start + example["fact_tokens"]]
        if bits.numel() > 0:  # empty if the answer ended before any decode step
            fact_kept.append((bits == FULL_PRECISION).float().mean().item())
        if out["token_ids"]:
            decode_ms.append(1000 * out["decode_seconds_per_token"])
        if example["context_length"] == max(PASSKEY_LENGTHS):
            kv_mib.append(cache.stored_bytes() / 2**20)

    average = lambda values: sum(values) / len(values) if values else float("nan")
    num_found = sum(sum(hits) for hits in found.values())
    return {
        "passkey_found": 100 * num_found / len(examples),
        "passkey_exact_match": 100 * sum(exact) / len(examples),
        "passkey_found_by_cell": {cell: 100 * sum(hits) / len(hits) for cell, hits in found.items()},
        "passkey_sentence_kept": 100 * average(fact_kept),
        "decode_ms_per_token": average(decode_ms),
        "kv_mib_measured_at_7k": average(kv_mib),
    }


def run_hotpotqa(model, tokenizer, examples, make_cache) -> float:
    scores = []
    for prompt, answers in tqdm(examples, desc="hotpotqa"):
        out = generate(model, tokenizer, prompt, make_cache, max_new_tokens=TASKS["hotpotqa"]["max_tokens"],
                       stop_token_ids=stop_token_ids(tokenizer))
        scores.append(score_prediction("hotpotqa", out["text"], answers))
    return 100 * sum(scores) / len(scores)


def save(results: list[dict], smoke: bool) -> None:
    if smoke:
        return
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "settings": {
                "passkey": {"lengths": PASSKEY_LENGTHS, "positions": TARGET_POSITIONS,
                            "examples_per_cell": PASSKEY_PER_CELL, "max_new_tokens": OUTPUT_TOKENS},
                "hotpotqa": {"examples": HOTPOTQA_EXAMPLES, "input_budget": HOTPOTQA_BUDGET,
                             "max_new_tokens": TASKS["hotpotqa"]["max_tokens"]},
                "window": WINDOW, "num_sinks": NUM_SINKS, "kv_low_bits": LOW_BITS,
                "kv_group_size": GROUP_SIZE or "head_dim", "key_norm_checkpoint": KEY_NORM_CKPT,
                "dtype": "float16", "decoding": "greedy",
            },
            "results": results,
        }, f, indent=2)


def print_table(results: list[dict]) -> None:
    print(f"\n{'Weights':30s} {'KV cache':31s} {'Passkey':>8s} {'Exact':>6s} {'Fact FP':>8s} "
          f"{'HotpotQA':>9s} {'ms/tok':>7s} {'KV MiB@7K':>10s}")
    for r in results:
        print(f"{r['weights']:30s} {r['kv_cache']:31s} {r['passkey_found']:7.1f}% {r['passkey_exact_match']:5.1f}% "
              f"{r['passkey_sentence_kept']:7.0f}% {r['hotpotqa_f1']:9.1f} {r['decode_ms_per_token']:7.1f} "
              f"{r['kv_mib_measured_at_7k']:10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="quick check: FP16 weights only, 2 passkey + 2 HotpotQA examples, nothing saved")
    args = parser.parse_args()
    weights = dict(list(WEIGHTS.items())[:1]) if args.smoke else WEIGHTS

    results, passkey, hotpotqa = [], None, None
    for weights_name, (path, weight_gb) in weights.items():
        model, tokenizer = load_model(path)
        if passkey is None:  # every checkpoint uses the same Llama-3 tokenizer
            if args.smoke:
                passkey = passkey_examples(tokenizer, per_cell=1, positions=(0.5,))
                hotpotqa = hotpotqa_examples(tokenizer, num_examples=2)
            else:
                passkey, hotpotqa = passkey_examples(tokenizer), hotpotqa_examples(tokenizer)
            print(f"{len(passkey)} passkey examples, {len(hotpotqa)} HotpotQA examples")

        # One untimed run with a quantizing cache, so the GPU's start-up cost (first kernel
        # launches, memory allocation) doesn't land in the first row's decode time
        warmup_cache = lambda: RealQuantKVCache(
            PositionScorer(), budget=0.0, low_bits=LOW_BITS, group_size=GROUP_SIZE,
            window=WINDOW, num_sinks=NUM_SINKS, config=model.config,
        )
        generate(model, tokenizer, passkey[-1]["prompt_token_ids"], warmup_cache, max_new_tokens=8, stop_token_ids=[])

        for kv_name, (make_scorer, budget) in KV_SETTINGS.items():
            scorer = make_scorer()
            make_cache = lambda: RealQuantKVCache(
                scorer, budget=budget, low_bits=LOW_BITS, group_size=GROUP_SIZE,
                window=WINDOW, num_sinks=NUM_SINKS, config=model.config,
            )
            row = {"weights": weights_name, "kv_cache": kv_name, "weight_gb": weight_gb}
            row.update(run_passkey(model, tokenizer, passkey, make_cache))
            row["hotpotqa_f1"] = run_hotpotqa(model, tokenizer, hotpotqa, make_cache)
            results.append(row)
            print(f"{weights_name} + {kv_name}: passkey {row['passkey_found']:.1f}%, "
                  f"HotpotQA F1 {row['hotpotqa_f1']:.1f}")
            save(results, args.smoke)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print_table(results)
    if not args.smoke:
        print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()