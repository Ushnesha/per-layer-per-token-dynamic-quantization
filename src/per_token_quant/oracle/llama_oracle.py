"""Oracle and signals for Llama-3-8B-Instruct, in the same format as the SmolLM-360M files.

Writes two files that the existing analysis and training scripts can read by changing paths:
    results/signals_llama3_8b_instruct_20x512.pt                (same keys as signals_run.py)
    results/oracle_kl_llama3_8b_instruct_int4_gfull_20x512.pt   (same keys as oracle.py)

Why the Instruct model: the passkey and LongBench prompts are wrapped in its chat template,
and the linear model has to be trained on the key norms of the model it will run on.

Needs:
    - An A100 80GB. In FP32 the model alone is 32 GB.
    - Access to the gated Llama-3 repo. Log in once on Sol with `huggingface-cli login`.
      Never put the token in code.
    - A few hours for the oracle: 512 forward passes per group of chunks. The script prints
      the time per group, and saves each group as it finishes. If the job gets killed, rerun
      it and the finished groups are skipped.

Run from the same folder as oracle.py.
"""

import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_wikitext
from .quant import FULL_PRECISION, FakeQuantKVCache
from .signals import compute_signals

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
SHORT_NAME = "llama3_8b_instruct"
DTYPE = torch.float32  # FP32 so the oracle's tiny KL values aren't buried in rounding noise
NUM_CHUNKS = 20
CHUNK_SIZE = 512
BITS = 4
GROUP_SIZE = None  # None = one group over the whole head_dim, as in the SmolLM oracle
CHUNKS_PER_GROUP = 5  # Llama's 128K vocabulary makes the logits big, so run 5 chunks at a time

group_name = "full" if GROUP_SIZE is None else GROUP_SIZE
ORACLE_PATH = f"results/oracle_kl_{SHORT_NAME}_int{BITS}_g{group_name}_{NUM_CHUNKS}x{CHUNK_SIZE}.pt"
SIGNALS_PATH = f"results/signals_{SHORT_NAME}_{NUM_CHUNKS}x{CHUNK_SIZE}.pt"


# The same three helpers as oracle.py. They're copied, not imported, because importing
# oracle.py would run the whole SmolLM oracle.
@torch.no_grad()
def next_token_log_probs(model, chunks: torch.Tensor, cache=None) -> torch.Tensor:
    """Log-probabilities for the next token at every position: [chunks, seq_len, vocab], in float32."""
    logits = model(chunks, past_key_values=cache, use_cache=True).logits
    return F.log_softmax(logits.float(), dim=-1)


def single_token_bits(seq_len: int, j: int, bits: int) -> torch.Tensor:
    """Bit map with token j at `bits` and every other token at full precision."""
    bit_map = torch.full((seq_len,), FULL_PRECISION)
    bit_map[j] = bits
    return bit_map


def kl_per_position(ref_log_probs: torch.Tensor, pert_log_probs: torch.Tensor) -> torch.Tensor:
    """KL(reference || perturbed) at every position, summed over the vocabulary: [chunks, seq_len]."""
    return F.kl_div(input=pert_log_probs, target=ref_log_probs, reduction="none", log_target=True).sum(dim=-1)


def oracle_kl(model, chunks: torch.Tensor) -> torch.Tensor:
    """KL grid [chunks, j, t] for one group: quantize token j alone, measure KL at every position t."""
    seq_len = chunks.shape[1]
    ref_log_probs = next_token_log_probs(model, chunks)
    rows = []
    for j in range(seq_len):
        cache = FakeQuantKVCache(single_token_bits(seq_len, j, BITS), group_size=GROUP_SIZE, config=model.config)
        rows.append(kl_per_position(ref_log_probs, next_token_log_probs(model, chunks, cache)).cpu())
        if (j + 1) % 64 == 0:
            print(f"  token {j + 1}/{seq_len}", flush=True)
    return torch.stack(rows, dim=1)


def part_path(group_index: int) -> str:
    return ORACLE_PATH.replace(".pt", f"_part{group_index}.pt")


def main():
    torch.set_float32_matmul_precision("highest")  # real FP32 matmuls, no TF32 shortcuts
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Eager attention, because the signals need the attention weights
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=DTYPE, attn_implementation="eager")
    model = model.to(device).eval()

    chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[:NUM_CHUNKS].to(device)
    groups = chunks.split(CHUNKS_PER_GROUP)
    os.makedirs(os.path.dirname(ORACLE_PATH), exist_ok=True)

    # Signals first: they take about a minute, so a broken setup shows up before the long oracle run
    if os.path.exists(SIGNALS_PATH):
        print(f"Signals already saved at {SIGNALS_PATH}, skipping")
    else:
        parts = []
        for group in groups:
            parts.append({name: value.cpu() for name, value in compute_signals(model, group).items()})
        signals = {name: torch.cat([part[name] for part in parts], dim=1) for name in parts[0]}  # [layers, chunks, seq]
        torch.save({"model": MODEL_NAME, "num_chunks": NUM_CHUNKS, "chunk_size": CHUNK_SIZE, **signals}, SIGNALS_PATH)
        print(f"Saved signals to {SIGNALS_PATH}")

    # Oracle, one group at a time, saving each group as it finishes
    kl_parts = []
    for g, group in enumerate(groups):
        path = part_path(g)
        if os.path.exists(path):
            print(f"Group {g + 1}/{len(groups)} already done, loading {path}")
            kl_parts.append(torch.load(path))
            continue
        print(f"Group {g + 1}/{len(groups)}: chunks {g * CHUNKS_PER_GROUP}-{g * CHUNKS_PER_GROUP + len(group) - 1}")
        start = time.time()
        part = oracle_kl(model, group)
        torch.save(part, path)
        kl_parts.append(part)
        print(f"Group {g + 1} done in {(time.time() - start) / 60:.1f} min", flush=True)

    kl = torch.cat(kl_parts, dim=0)  # [chunks, j, t]
    torch.save(
        {
            "kl": kl,
            "model": MODEL_NAME,
            "bits": BITS,
            "group_size": GROUP_SIZE,
            "num_chunks": kl.shape[0],
            "chunk_len": kl.shape[-1],
            "dtype": str(DTYPE).removeprefix("torch."),
        },
        ORACLE_PATH,
    )
    for g in range(len(groups)):
        os.remove(part_path(g))
    print(f"Saved results to {ORACLE_PATH}")


if __name__ == "__main__":
    main()