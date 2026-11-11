import os

import torch
import torch.nn.functional as F

from data import load_wikitext
from models import get_model_and_tokenizer
from quant import FULL_PRECISION, FakeQuantKVCache

MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
DTYPE = torch.float32
NUM_CHUNKS = 20
CHUNK_SIZE = 512
BITS = 4
GROUP_SIZE = None  # None = one group over the whole head_dim

group_name = "full" if GROUP_SIZE is None else GROUP_SIZE
OUTPUT_PATH = f"results/oracle_kl_smollm360m_int{BITS}_g{group_name}_{NUM_CHUNKS}x{CHUNK_SIZE}.pt"


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


device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = get_model_and_tokenizer(model_name=MODEL_NAME, dtype=DTYPE, device=device)
chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[:NUM_CHUNKS].to(device)
seq_len = chunks.shape[1]

# Reference pass: nothing quantized
ref_log_probs = next_token_log_probs(model, chunks)

# Perturbed passes: quantize only token j, once for every position j
kl_rows = []
for j in range(seq_len):
    cache = FakeQuantKVCache(single_token_bits(seq_len, j, BITS), group_size=GROUP_SIZE, config=model.config)
    kl_rows.append(kl_per_position(ref_log_probs, next_token_log_probs(model, chunks, cache)))
    if (j + 1) % 50 == 0:
        print(f"Finished {j + 1}/{seq_len}")

kl = torch.stack(kl_rows, dim=1).cpu()  # [chunks, j, t]

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(
    {
        "kl": kl,
        "model": MODEL_NAME,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "num_chunks": chunks.shape[0],
        "chunk_len": seq_len,
        "dtype": str(DTYPE).removeprefix("torch."),
    },
    OUTPUT_PATH,
)
print(f"Saved results to {OUTPUT_PATH}")