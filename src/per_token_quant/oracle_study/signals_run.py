import torch
from collections import defaultdict
import os

from models import get_model_and_tokenizer
from signals import compute_signals
from data import load_wikitext

MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
CHUNK_SIZE = 512
BATCH_SIZE = 5
NUM_CHUNKS = 20
DTYPE = torch.float32
OUTPUT_PATH = f"results/signals_smollm360m_{NUM_CHUNKS}x{CHUNK_SIZE}.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = get_model_and_tokenizer(model_name=MODEL_NAME, dtype=DTYPE, device=device)
chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[:NUM_CHUNKS].to(device)

signals_res = defaultdict(list)
for start in range(0, NUM_CHUNKS, BATCH_SIZE):
    out = compute_signals(model, chunks[start:start+BATCH_SIZE, :])
    for signal, results in out.items():
        signals_res[signal].append(results.cpu())
signals_res = {signal: torch.cat(results, dim=1) for signal, results in signals_res.items()}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(
    {
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "num_chunks": NUM_CHUNKS,
        "dtype": str(DTYPE).removeprefix("torch."),
        **signals_res
    },
    OUTPUT_PATH,
)
