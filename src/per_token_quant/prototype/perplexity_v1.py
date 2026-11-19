"""WikiText-2 perplexity on Llama-3-8B-Instruct with prototype v1 (real 2-bit KV storage).

Each chunk is streamed through the cache 64 tokens at a time, so tokens leave the window and
get quantized the way they would while generating.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle.data import load_wikitext
from oracle.metrics import perplexity
from oracle.quant import FULL_PRECISION
from prototype.adaptive_kv import KeyNormLinearScorer, PositionScorer, RandomScorer
from prototype.real_kv import RealQuantKVCache

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = "l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
CHUNK_SIZE = 2048
CHUNKS = slice(5, 25)  # 20 chunks; the first 5 hold the text the oracle and linear model were trained on
LOW_BITS, GROUP_SIZE = 2, None  # INT2 group 128, as in passkey and LongBench
BUDGET = 0.1
STRIDE = 64

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[CHUNKS]


def kept_fraction(cache):
    """Share of the decided tokens (sinks excluded) that the policy kept in full precision."""
    decided = cache.bits[cache.num_sinks:]
    return (decided == FULL_PRECISION).float().mean().item() if decided.numel() else 1.0


settings = {
    "full precision": (PositionScorer, 1.0),
    "static INT2": (PositionScorer, 0.0),
    f"key norm @ {BUDGET:.0%}": (lambda: KeyNormLinearScorer(CKPT), BUDGET),
    f"random @ {BUDGET:.0%}": (lambda: RandomScorer(0), BUDGET),
}
for name, (make_scorer, budget) in settings.items():
    scorer = make_scorer()
    kept, previous = [], []

    def make_cache():
        if previous:  # record the finished chunk's cache, then let it go
            kept.append(kept_fraction(previous.pop()))
        cache = RealQuantKVCache(scorer, budget=budget, low_bits=LOW_BITS, group_size=GROUP_SIZE, config=model.config)
        previous.append(cache)
        return cache

    ppl = perplexity(model, chunks, batch_size=1, cache_factory=make_cache, stride=STRIDE, show_progress=True)
    kept.append(kept_fraction(previous.pop()))
    print(f"{name:20s} perplexity {ppl:.3f} | tokens kept in full precision {100 * sum(kept) / len(kept):.1f}%")