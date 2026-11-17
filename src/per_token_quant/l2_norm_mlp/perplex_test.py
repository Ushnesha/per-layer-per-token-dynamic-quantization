import torch
import os

from .policies import policy_perplexity, key_norm_linear_scores, windowed_attention_scores, position_scores, random_scores
from oracle.models import get_model_and_tokenizer
from oracle.data import load_wikitext

MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 512
FULL_PRECISION_BASELINE = 17.347
INT2_BASELINE = 18.478
CKPT_PATH = "./l2_norm_mlp/results/key_norm_linear_smollm360m.pt"
OUTPUT_PATH = "./l2_norm_mlp/results/perplexity_smollm360m.pt"

model, tokenizer = get_model_and_tokenizer(model_name=MODEL_NAME, dtype=DTYPE, device=DEVICE)
chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[20:, :]

# Baseline 
# # full precision
# ppl1 = policy_perplexity(
#     model,
#     chunks,
#     score_fn=None,
#     budget=1.0,
#     low_bits=16,
#     batch_size=8,
#     show_progress=True,
# )
# # Uniform INT2, Group size 32
# ppl2 = policy_perplexity(
#     model,
#     chunks,
#     score_fn=None,
#     budget=0.0,
#     low_bits=2,
#     group_size=32,
#     batch_size=8,
#     show_progress=True,
# )
# # Uniform INT3, Group size 32
# ppl3 = policy_perplexity(
#     model,
#     chunks,
#     score_fn=None,
#     budget=0.0,
#     low_bits=3,
#     group_size=32,
#     batch_size=8,
#     show_progress=True,
# )

# print(f"Full Precision: {ppl1:.3f} | INT2 g32: {ppl2:.3f} | INT3 g32: {ppl3:.3f}")

policies = {
    "linear (key norm)": lambda: key_norm_linear_scores(CKPT_PATH),
    "attention (window 128)": lambda: windowed_attention_scores(128),
    "position": lambda: position_scores(),
    "random": lambda: random_scores(0),
}

budgets = [0.1, 0.2]
res = {}
for budget in budgets:
    for policy_name, make_policy in policies.items():
        ppl = policy_perplexity(
            model,
            chunks,
            make_policy(),
            budget=budget,
            low_bits=2,
            group_size=32,
            batch_size=8,
            show_progress=True,
        )
        print(f"{policy_name} ({budget}): {ppl:.3f}")
        res[f"{policy_name}_{budget}"] = ppl

# Results table
gap = INT2_BASELINE - FULL_PRECISION_BASELINE
print("\nPerplexity (lower is better), SmolLM-360M, INT2 g32")
print(f"{'full precision':25s} {FULL_PRECISION_BASELINE:.3f}")
print(f"{'uniform INT2':25s} {INT2_BASELINE:.3f}")
for budget in budgets:
    print(f"\nBudget: {budget:.0%} kept in full precision")
    for policy_name in policies:
        ppl = res[f"{policy_name}_{budget}"]
        recovered = (INT2_BASELINE - ppl) / gap
        print(f"{policy_name:25s} {ppl:.3f}  (recovers {recovered:.0%} of the INT2 gap)")

# Saving
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(
    {
        "results": res,
        "baselines": {"full_precision": FULL_PRECISION_BASELINE, "uniform_int2": INT2_BASELINE},
        "settings": {
            "llm": MODEL_NAME,
            "dtype": str(DTYPE),
            "chunk_size": CHUNK_SIZE,
            "chunks": "all chunks after the first 20",
            "num_chunks": chunks.shape[0],
            "low_bits": 2,
            "group_size": 32,
            "budgets": budgets,
            "window": 128,  # policy_perplexity defaults
            "num_sinks": 4,
            "stride": 64,
            "checkpoint": CKPT_PATH,
        },
    },
    OUTPUT_PATH,
)
print(f"\nSaved to {OUTPUT_PATH}")