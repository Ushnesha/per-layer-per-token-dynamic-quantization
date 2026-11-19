"""Checks the weight loader against Ushnesha's vLLM perplexities before building on it.

Loads each checkpoint, computes WikiText-2 perplexity on the same 20 chunks of 2,048 tokens with
no KV-cache quantization, and prints how much each model is above FP16, next to the increase she
measured in vLLM. The absolute numbers differ from hers (20 windows instead of all 141, different
BOS handling), but the relative increase should be close if the loader is right.

Run from the repo root:
    python -m src.end_to_end.check_weights
"""

import gc

import torch

from src.end_to_end.quantized_weights import load_model
from oracle.data import load_wikitext
from oracle.metrics import perplexity

# name: (checkpoint, Ushnesha's vLLM WikiText-2 perplexity)
CHECKPOINTS = {
    "FP16": ("models/llama-3-8b-instruct", 8.2504),
    "GPTQ W4 (static)": ("models/gptq/llama3-8b-instruct-w4a16", 8.977),
    "Mixed ref-kl-5.50": ("models/mixed/llama3-8b-instruct-ref-kl-5.50", 8.5861),
}
NUM_CHUNKS = 20

ours = {}
for name, (path, _) in CHECKPOINTS.items():
    model, tokenizer = load_model(path)
    chunks = load_wikitext(tokenizer=tokenizer, chunk_size=2048)[:NUM_CHUNKS]
    ours[name] = perplexity(model, chunks, show_progress=True)
    print(f"{name}: perplexity {ours[name]:.3f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

fp16_ours, fp16_vllm = ours["FP16"], CHECKPOINTS["FP16"][1]
print("\nIncrease over FP16 (ours on 20 chunks vs. Ushnesha's vLLM on the full test set):")
for name, (_, vllm) in CHECKPOINTS.items():
    print(f"{name:20s} ours {ours[name]:.3f} (+{100 * (ours[name] / fp16_ours - 1):.1f}%) | "
          f"vLLM {vllm:.3f} (+{100 * (vllm / fp16_vllm - 1):.1f}%)")