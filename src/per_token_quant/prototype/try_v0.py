import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from oracle.data import load_wikitext
from prototype.adaptive_kv import AdaptiveKVCache, KeyNormLinearScorer, generate

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = "l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# A long prompt: ~2,000 tokens of WikiText the linear model never saw, then a question about it
article = tokenizer.decode(load_wikitext(tokenizer=tokenizer, chunk_size=2048)[5], skip_special_tokens=True)
messages = [{"role": "user", "content": article + "\n\nSummarize the text above in three sentences."}]
prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

scorer = KeyNormLinearScorer(CKPT)
for budget in (1.0, 0.1, 0.0):  # full precision, key-norm policy at 10%, uniform INT2
    make_cache = lambda: AdaptiveKVCache(scorer, budget=budget, low_bits=2, group_size=32, config=model.config)
    torch.cuda.reset_peak_memory_stats()
    out = generate(model, tokenizer, prompt, make_cache, max_new_tokens=100)
    report = out["cache"].memory_report()

    print(f"\n--- budget {budget:.0%} ---")
    print(out["text"])
    print(f"prompt tokens: {len(prompt)} | full precision: {report['full_precision_tokens']} | low-bit: {report['low_bit_tokens']}")
    print(f"KV compression: {report['compression']:.2f}x | average bits per value: {report['average_bits_per_value']:.2f}")
    print(f"prefill {out['prefill_seconds']:.2f} s | decode {out['decode_seconds_per_token'] * 1000:.1f} ms/token | "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")