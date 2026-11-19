"""Checks the weight loader against Ushnesha's vLLM perplexities before building on it.

Loads each checkpoint, computes WikiText-2 perplexity on the same 20 chunks of 2,048 tokens with
no KV-cache quantization, and prints how much each model is above FP16, next to the increase she
measured in vLLM. The absolute numbers differ from hers (20 windows instead of all 141, different
BOS handling), but the relative increase should be close if the loader is right.

Run from the repo root:
    python -m src.end_to_end.check_weights        # every checkpoint
    python -m src.end_to_end.check_weights awq    # FP16 plus the checkpoints with "awq" in their name
"""

import gc
import sys

import torch

from src.end_to_end.quantized_weights import load_model
from oracle.data import load_wikitext
from oracle.metrics import perplexity

# name: (checkpoint, Ushnesha's vLLM WikiText-2 perplexity from results/*.json)
CHECKPOINTS = {
    "FP16": ("models/llama-3-8b-instruct", 8.2504),
    "GPTQ W8 (static)": ("models/gptq/llama3-8b-instruct-w8a16", 8.2569),
    "GPTQ W4 (static)": ("models/gptq/llama3-8b-instruct-w4a16", 8.977),
    "Mixed GPTQ ref-kl-5.50": ("models/mixed/llama3-8b-instruct-ref-kl-5.50", 8.5861),
    "AWQ W4 (static)": ("models/awq/llama3-8b-instruct-w4a16", 8.5912),
    "Mixed AWQ ref-kl-5.50": ("models/mixed/awq/llama3-8b-instruct-ref-kl-5.50", 8.4844),
}
NUM_CHUNKS = 20

# Optional filter: only the checkpoints whose name contains the word given (FP16 always runs)
word = sys.argv[1].lower() if len(sys.argv) > 1 else ""
selected = {name: v for name, v in CHECKPOINTS.items() if name == "FP16" or word in name.lower()}

ours = {}
for name, (path, _) in selected.items():
    model, tokenizer = load_model(path)
    chunks = load_wikitext(tokenizer=tokenizer, chunk_size=2048)[:NUM_CHUNKS]
    ours[name] = perplexity(model, chunks, show_progress=True)
    print(f"{name}: perplexity {ours[name]:.3f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

fp16_ours, fp16_vllm = ours["FP16"], CHECKPOINTS["FP16"][1]
print("\nIncrease over FP16 (ours on 20 chunks vs. Ushnesha's vLLM on the full test set):")
for name, (_, vllm) in selected.items():
    print(f"{name:24s} ours {ours[name]:.3f} ({100 * (ours[name] / fp16_ours - 1):+.1f}%) | "
          f"vLLM {vllm:.3f} ({100 * (vllm / fp16_vllm - 1):+.1f}%)")