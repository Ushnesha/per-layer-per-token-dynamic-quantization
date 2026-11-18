"""Smoke test for prototype v1 on Llama-3-8B-Instruct: the same prompt as try_v0, real low-bit storage.

For each budget it prints the answer and three memory numbers for the KV cache:
    - stored: the bytes the cache's two piles really hold
    - predicted: what v0's memory_report() said a real implementation would use
    - measured on GPU: how much GPU memory is still allocated after generation, i.e. the cache
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle.data import load_wikitext
from prototype.adaptive_kv import KeyNormLinearScorer, generate
from prototype.real_kv import RealQuantKVCache

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = "l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
MIB = 2**20

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# The same long prompt as try_v0: ~2,000 tokens of WikiText, then a question about it
article = tokenizer.decode(load_wikitext(tokenizer=tokenizer, chunk_size=2048)[5], skip_special_tokens=True)
messages = [{"role": "user", "content": article + "\n\nSummarize the text above in three sentences."}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

scorer = KeyNormLinearScorer(CKPT)
for budget in (1.0, 0.1, 0.0):  # full precision, key-norm policy at 10%, uniform INT2
    make_cache = lambda: RealQuantKVCache(scorer, budget=budget, low_bits=2, group_size=32, config=model.config)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = generate(model, tokenizer, prompt, make_cache, max_new_tokens=100)
    torch.cuda.synchronize()
    measured = torch.cuda.memory_allocated() - before  # the cache is the main thing still alive
    cache = out["cache"]
    report = cache.memory_report()

    print(f"\n--- budget {budget:.0%} ---")
    print(out["text"])
    print(f"full precision tokens: {report['full_precision_tokens']} | low-bit: {report['low_bit_tokens']}")
    print(
        f"KV cache: stored {cache.stored_bytes() / MIB:.1f} MiB | predicted {report['kv_bytes'] / MIB:.1f} MiB | "
        f"measured on GPU {measured / MIB:.1f} MiB | full-precision cache would be {report['fp16_kv_bytes'] / MIB:.1f} MiB"
    )
    print(f"decode {out['decode_seconds_per_token'] * 1000:.1f} ms/token")
    del out, cache