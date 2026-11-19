"""End-to-end pipeline: per-layer weight quantization x per-token KV-cache quantization.

For every weight checkpoint and every KV-cache setting, it streams WikiText-2 through the model
and reports perplexity, plus the memory the weights and the KV cache would take in a real
deployment.

    Weights (Ushnesha, src/adaptive): FP16, static GPTQ W8 and W4, static AWQ W4, and per-layer
        mixed GPTQ and AWQ (ref-kl-5.50)
    KV cache (Ernesto, src/per_token_quant, prototype v1): full precision, static INT2,
        key-norm policy keeping 10% of tokens in full precision

How each part is measured:
    - Perplexity: 20 chunks of 2,048 WikiText-2 tokens (skipping the first 5, which the key-norm
      model was trained on), streamed 64 tokens at a time so tokens leave the 128-token window
      and get quantized the way they would while generating.
    - Weight memory: the checkpoint size from Ushnesha's results. The weights are unpacked to FP16
      here (see quantized_weights.py), so the GPU doesn't show the real saving.
    - Static GPTQ W4: Ushnesha's vLLM result is 8.977 at full-precision KV, but this loader gives
      about 44 for the same folder (her other checkpoints match). Still being checked with her.
    - KV memory: for one 4,096-token sequence, counted the same way as RealQuantKVCache stores it
      (FP16 tokens at 16 bits; INT2 tokens at 2 bits plus an FP16 scale and zero point per group
      of 128 values).

Run from the repo root:
    python -m src.end_to_end.run_e2e        # every checkpoint, about 15 minutes on an A100
    python -m src.end_to_end.run_e2e awq    # only the checkpoints with "awq" in their name

With a filter, only those checkpoints run again. The saved results for the others are kept, and
the table shows all of them.
"""

import gc
import json
import os
import sys

import torch

from src.end_to_end.quantized_weights import load_model
from oracle.data import load_wikitext
from oracle.metrics import perplexity
from prototype.adaptive_kv import KeyNormLinearScorer, PositionScorer
from prototype.real_kv import RealQuantKVCache

KEY_NORM_CKPT = "src/per_token_quant/l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
OUTPUT_PATH = "results/end_to_end/e2e_wikitext2.json"

# name: (checkpoint, weight storage in GB from Ushnesha's results/*.json)
WEIGHTS = {
    "FP16": ("models/llama-3-8b-instruct", 16.061),
    "GPTQ W8 (static)": ("models/gptq/llama3-8b-instruct-w8a16", 9.25),
    "GPTQ W4 (static)": ("models/gptq/llama3-8b-instruct-w4a16", 5.733),
    "Mixed GPTQ ref-kl-5.50 (per-layer)": ("models/mixed/llama3-8b-instruct-ref-kl-5.50", 5.499),
    "AWQ W4 (static)": ("models/awq/llama3-8b-instruct-w4a16", 5.728),
    "Mixed AWQ ref-kl-5.50 (per-layer)": ("models/mixed/awq/llama3-8b-instruct-ref-kl-5.50", 5.485),
}
RENAMED = {"Mixed ref-kl-5.50 (per-layer)": "Mixed GPTQ ref-kl-5.50 (per-layer)"}  # name in older results
# name: (scorer factory, fraction of the tokens leaving the window kept in full precision)
KV_SETTINGS = {
    "full precision": (PositionScorer, 1.0),
    "static INT2": (PositionScorer, 0.0),
    "key norm, 10% kept (per-token)": (lambda: KeyNormLinearScorer(KEY_NORM_CKPT), 0.1),
}
LOW_BITS, GROUP_SIZE = 2, None  # INT2 with one group per head (128 values)
WINDOW, NUM_SINKS, STRIDE = 128, 4, 64
CHUNK_SIZE, CHUNKS = 2048, slice(5, 25)
MEMORY_CONTEXT = 4096  # KV memory is reported for one sequence of this many tokens


def kv_mib(config, kept_fraction: float, context: int = MEMORY_CONTEXT) -> float:
    """KV-cache memory for one `context`-token sequence, the way RealQuantKVCache stores it."""
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    values_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim  # K + V, all layers
    group = head_dim if GROUP_SIZE is None else GROUP_SIZE
    competing = max(context - NUM_SINKS - WINDOW, 0)
    num_low = round(competing * (1 - kept_fraction))
    num_full = context - num_low
    full_bytes = num_full * values_per_token * 2
    low_bytes = num_low * (values_per_token * LOW_BITS / 8 + values_per_token // group * 2 * 2)
    return (full_bytes + low_bytes) / 2**20


def previous_results(selected) -> list[dict]:
    """Saved rows for the checkpoints that aren't being run again."""
    if not os.path.exists(OUTPUT_PATH):
        return []
    with open(OUTPUT_PATH) as f:
        rows = json.load(f)["results"]
    for row in rows:
        row["weights"] = RENAMED.get(row["weights"], row["weights"])
    return [row for row in rows if row["weights"] in WEIGHTS and row["weights"] not in selected]


def in_order(rows: list[dict]) -> list[dict]:
    weights, kv = list(WEIGHTS), list(KV_SETTINGS)
    return sorted(rows, key=lambda r: (weights.index(r["weights"]), kv.index(r["kv_cache"])))


# Optional filter: only the checkpoints whose name contains the word given
word = sys.argv[1].lower() if len(sys.argv) > 1 else ""
selected = {name: v for name, v in WEIGHTS.items() if word in name.lower()}
if not selected:
    sys.exit(f"No checkpoint name contains '{word}'. Names: {list(WEIGHTS)}")
kept = previous_results(selected)

results = []
for weights_name, (path, weight_gb) in selected.items():
    model, tokenizer = load_model(path)
    chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[CHUNKS]

    for kv_name, (make_scorer, budget) in KV_SETTINGS.items():
        scorer = make_scorer()
        make_cache = lambda: RealQuantKVCache(
            scorer, budget=budget, low_bits=LOW_BITS, group_size=GROUP_SIZE,
            window=WINDOW, num_sinks=NUM_SINKS, config=model.config,
        )
        ppl = perplexity(model, chunks, cache_factory=make_cache, stride=STRIDE, show_progress=True)
        kv = kv_mib(model.config, budget)
        results.append({
            "weights": weights_name, "kv_cache": kv_name, "perplexity": ppl,
            "weight_gb": weight_gb, "kv_mib_at_4k": kv, "total_gb": weight_gb + kv / 1024,
        })
        print(f"{weights_name} + {kv_name}: perplexity {ppl:.3f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

results = in_order(kept + results)
print(f"\n{'Weights':36s} {'KV cache':32s} {'PPL':>7s} {'Weights GB':>11s} {'KV MiB @4K':>11s} {'Total GB':>9s}")
for r in results:
    print(f"{r['weights']:36s} {r['kv_cache']:32s} {r['perplexity']:7.3f} {r['weight_gb']:11.2f} "
          f"{r['kv_mib_at_4k']:11.1f} {r['total_gb']:9.2f}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump({
        "settings": {
            "dataset": "wikitext-2-raw-v1 test", "chunk_size": CHUNK_SIZE, "chunks": "5-24",
            "stride": STRIDE, "window": WINDOW, "num_sinks": NUM_SINKS,
            "kv_low_bits": LOW_BITS, "kv_group_size": GROUP_SIZE or "head_dim",
            "key_norm_checkpoint": KEY_NORM_CKPT, "dtype": "float16",
            "memory_context_tokens": MEMORY_CONTEXT,
        },
        "results": results,
    }, f, indent=2)
print(f"\nSaved to {OUTPUT_PATH}")