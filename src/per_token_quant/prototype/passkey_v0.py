import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(".."))  # src/, so Aditya's evaluation folder can be imported

from evaluation.benchmark.retrieval import CONTEXT_LENGTHS, TARGET_POSITIONS, SEED, build_example, exact_match
from evaluation.benchmark.instruction_prompt import stop_token_ids
from prototype.adaptive_kv import AdaptiveKVCache, KeyNormLinearScorer, PositionScorer, RandomScorer, generate

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = "l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
LOW_BITS = 2
GROUP_SIZE = 32

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# Calibration: full precision vs. uniform INT2. The scorer doesn't matter at budget 1.0 or 0.0.
for name, budget in (("full precision", 1.0), ("uniform INT2", 0.0)):
    make_cache = lambda: AdaptiveKVCache(PositionScorer(), budget=budget, low_bits=LOW_BITS,
                                         group_size=GROUP_SIZE, config=model.config)
    per_cell, overall = run_passkey(model, make_cache)
    print_accuracy(name, per_cell, overall)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
LENGTHS = (2048, 7168)
EXAMPLES_PER_CELL = 10

examples = []
for length in LENGTHS:
    length_index = CONTEXT_LENGTHS.index(length)  # same seeds as Aditya's full benchmark
    for position_index, position in enumerate(TARGET_POSITIONS):
        for index in range(EXAMPLES_PER_CELL):
            seed = SEED + length_index * 100_000 + position_index * 1000 + index
            example = build_example(tokenizer, length, position, seed)
            example["context_length"] = length
            examples.append(example)

print(f"{len(examples)} examples")

def run_passkey(model, make_cache):
    """Passkey accuracy (%) for each (length, position) cell, and overall."""
    outcomes = {}
    for example in tqdm(examples, desc="passkey"):
        out = generate(model, tokenizer, example["prompt_token_ids"], make_cache, max_new_tokens=32,
                       stop_token_ids=stop_token_ids(tokenizer), stop_text="\n")
        cell = (example["context_length"], example["target_position"])
        outcomes.setdefault(cell, []).append(exact_match(out["text"], example["expected"]))
    per_cell = {cell: 100 * sum(hits) / len(hits) for cell, hits in sorted(outcomes.items())}
    overall = 100 * sum(sum(hits) for hits in outcomes.values()) / len(examples)
    return per_cell, overall


def print_accuracy(name, per_cell, overall):
    cells = " | ".join(f"{length} @ {position:.0%}: {acc:.0f}%" for (length, position), acc in per_cell.items())
    print(f"{name:25s} overall {overall:5.1f}% | {cells}")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# Calibration: full precision vs. uniform INT2. The scorer doesn't matter at budget 1.0 or 0.0.
for name, budget in (("full precision", 1.0), ("uniform INT2", 0.0)):
    make_cache = lambda: AdaptiveKVCache(PositionScorer(), budget=budget, low_bits=LOW_BITS,
                                         group_size=GROUP_SIZE, config=model.config)
    per_cell, overall = run_passkey(model, make_cache)
    print_accuracy(name, per_cell, overall)