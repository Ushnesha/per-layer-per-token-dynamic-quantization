# End-to-end pipeline: per-layer weights + per-token KV cache

This folder puts the two halves of the project together and tests them as one system:

- **Weights:** Ushnesha's checkpoints from `src/adaptive`: static GPTQ and AWQ, plus per-layer mixed precision (GPTQ and AWQ versions) where each layer gets 4 or 8 bits.
- **KV cache:** Ernesto's per-token cache from `src/per_token_quant` (prototype v1, real 2-bit storage). It keeps the most important 10% of tokens at full precision, picked by the learned key-norm model, and stores the rest in INT2.
- **Evaluation:** WikiText-2 perplexity, plus passkey retrieval and LongBench HotpotQA built and scored with Aditya's code in `src/evaluation/benchmark`.

## How to run

**Packages:** Python 3.10+, `torch`, `transformers` 5.x, `datasets`, `huggingface_hub`, `safetensors` and `tqdm`. The team's `requirements.txt` already includes all of them. vLLM, GPTQModel and compressed-tensors aren't needed, because `quantized_weights.py` unpacks the weights itself.

Run everything **from the repo root**, on a GPU:

```bash
python -m src.end_to_end.check_weights          # checks the weight loader against Ushnesha's vLLM numbers
python -m src.end_to_end.run_e2e                # WikiText-2 perplexity, every weights x KV combination (~15 min)
python -m src.end_to_end.run_e2e_tasks --smoke  # 2-3 minute check of the long-context tests
python -m src.end_to_end.run_e2e_tasks          # passkey + HotpotQA, every combination (~30 min)
```

All three take an optional word to run only some checkpoints, e.g. `python -m src.end_to_end.run_e2e awq` runs only the checkpoints with "awq" in their name. The saved results for the others are kept.

`models/` in the repo root must link to the shared checkpoints. Llama-3 is gated, so log in once with `huggingface-cli login` (never put the token in code). Results are saved to `results/end_to_end/`.

## Files

| File | What it does |
| --- | --- |
| `__init__.py` | Puts `src/per_token_quant` on the import path, so its `oracle` and `prototype` packages import |
| `quantized_weights.py` | `load_model(path)` loads a checkpoint into Hugging Face and unpacks its quantized layers to FP16, so the KV cache can be swapped in. Reads the plain FP16 model, GPTQ v1 (static and mixed GPTQ), AutoAWQ GEMM (static AWQ) and compressed-tensors pack-quantized (mixed AWQ) |
| `check_weights.py` | Perplexity of each checkpoint, compared with Ushnesha's vLLM results |
| `run_e2e.py` | WikiText-2 perplexity for every weights x KV-cache combination, plus weight and KV memory |
| `run_e2e_tasks.py` | Passkey retrieval and HotpotQA for every combination, using Aditya's examples, prompts and scorers |

## Setup

- **Model:** Llama-3-8B-Instruct, FP16, greedy decoding, batch size 1.
- **Weights:** FP16, static GPTQ W8 and W4, static AWQ W4, and the mixed `ref-kl-5.50` allocation (5.5 bits per weight on average) in both GPTQ and AWQ form. All use group size 128. GPTQ is symmetric; AWQ's 4-bit layers have zero points.
- **KV cache:** 4 sink tokens and the latest 128 tokens always stay at full precision. Older tokens are stored as:
  - **full precision:** nothing is quantized
  - **static INT2:** every token is INT2, with one scale and zero point per 128 values
  - **key norm, 10% kept:** the key-norm model keeps the top 10% in full precision; the rest are INT2
- **Perplexity:** 20 WikiText-2 chunks of 2,048 tokens (chunks 5–24, since the key-norm model was trained on the first ones), fed 64 tokens at a time so tokens actually get quantized.
- **Passkey:** Aditya's synthetic examples: 2,048- and 7,168-token prompts, passkey at 10/50/90% depth, 10 per cell, 60 in total. Same seeds as his benchmark.
- **HotpotQA:** the first 50 LongBench examples, trimmed to 7,168 tokens, token F1.

## Results

**WikiText-2 perplexity** (KV memory for one 4,096-token sequence)

| Weights | Weight GB | Full-precision KV | Static INT2 | Key norm, 10% kept |
| --- | --- | --- | --- | --- |
| FP16 | 16.06 | 8.464 | 22.078 | 19.625 |
| GPTQ W8 | 9.25 | 8.463 | 22.482 | 19.906 |
| GPTQ W4* | 5.73 | 36.012 | 117.807 | 109.220 |
| AWQ W4 | 5.73 | 8.892 | 28.054 | 24.223 |
| Mixed GPTQ ref-kl-5.50 | 5.50 | 8.784 | 24.981 | 21.761 |
| Mixed AWQ ref-kl-5.50 | 5.49 | 8.755 | 26.271 | 22.762 |
| **KV MiB at 4K** | | 512.0 | 86.2 | 128.7 |

**Passkey retrieval and HotpotQA** (passkey = the passkey appears in the answer)

| Weights | KV cache | Passkey | Exact match | Passkey sentence kept at FP | HotpotQA F1 | Decode ms/token | KV MiB at 7K |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | full precision | 100.0% | 78.3% | 100% | 40.8 | 27.2 | 896.5 |
| FP16 | static INT2 | 78.3% | 61.7% | 0% | 35.1 | 69.6 | 140.3 |
| FP16 | key norm, 10% | 100.0% | 78.3% | 97% | 34.3 | 68.6 | 215.9 |
| GPTQ W8 | full precision | 100.0% | 81.7% | 100% | 40.8 | 28.5 | 896.5 |
| GPTQ W8 | static INT2 | 83.3% | 70.0% | 0% | 33.0 | 72.5 | 140.2 |
| GPTQ W8 | key norm, 10% | 100.0% | 81.7% | 97% | 37.8 | 70.5 | 215.9 |
| GPTQ W4* | full precision | 100.0% | 85.0% | 100% | 34.2 | 26.6 | 896.5 |
| GPTQ W4* | static INT2 | 75.0% | 66.7% | 0% | 28.9 | 70.1 | 140.2 |
| GPTQ W4* | key norm, 10% | 100.0% | 85.0% | 94% | 26.4 | 69.8 | 215.9 |
| AWQ W4 | full precision | 100.0% | 81.7% | 100% | 37.7 | 28.5 | 896.4 |
| AWQ W4 | static INT2 | 83.3% | 68.3% | 0% | 31.7 | 73.1 | 140.2 |
| AWQ W4 | key norm, 10% | 100.0% | 81.7% | 93% | 27.1 | 72.9 | 215.9 |
| Mixed GPTQ ref-kl-5.50 | full precision | 100.0% | 51.7% | 100% | 39.3 | 26.4 | 896.9 |
| Mixed GPTQ ref-kl-5.50 | static INT2 | 73.3% | 36.7% | 0% | 30.3 | 70.2 | 140.3 |
| Mixed GPTQ ref-kl-5.50 | key norm, 10% | 96.7% | 50.0% | 95% | 28.8 | 69.0 | 216.0 |
| Mixed AWQ ref-kl-5.50 | full precision | 100.0% | 70.0% | 100% | 37.2 | 28.1 | 896.5 |
| Mixed AWQ ref-kl-5.50 | static INT2 | 80.0% | 60.0% | 0% | 28.1 | 73.8 | 140.3 |
| Mixed AWQ ref-kl-5.50 | key norm, 10% | 95.0% | 70.0% | 95% | 31.2 | 73.9 | 215.9 |

\* Static GPTQ W4 gives a perplexity of 36 here, while Ushnesha's vLLM result for the same folder is 8.977. The same loader matches all her other checkpoints, including AWQ W4, so this file is still being checked. Its numbers shouldn't be used yet.

## What the results show

- **The weight loader matches vLLM.** With full-precision KV, the increase over FP16 is close to Ushnesha's: W8 +0.0% (vLLM +0.1%), mixed GPTQ +3.8% (+4.1%), AWQ W4 +5.1% (+4.1%), mixed AWQ +3.4% (+2.8%). The ranking is the same as hers. The absolute numbers differ (8.46 here vs. 8.25 in vLLM) because the perplexity setup is different.
- **The per-token policy protects retrieval with every weight setup.** Static INT2 loses 17–27% of the passkeys. Keeping 10% by key norm brings it back to 95–100%, because it keeps 93–97% of the passkey sentence at full precision.
- **It doesn't help on HotpotQA.** Against static INT2, key norm ranges from −4.6 to +4.8 F1 across the six weight setups and averages out to about zero. Those differences are within noise for 50 examples. Key norm doesn't look at the question: it finds the passkey because it stands out from repeated filler, but in real documents what matters depends on what's being asked.
- **On WikiText-2, it closes about 18–20% of the INT2 gap** with every weight setup. In prototype v1, random at 10% did about as well (19.06 vs. 19.41 for key norm), so on short text most of that comes from keeping some tokens at full precision, not from which ones.
- **Weight and KV quantization add up, and the best weights depend on the KV cache.** Every quantized-weight model with INT2 KV is worse than either one alone. Mixed AWQ beats mixed GPTQ with full-precision KV (8.755 vs. 8.784) but loses once the KV cache is quantized (26.27 vs. 24.98 with INT2, 22.76 vs. 21.76 with key norm). Of the checkpoints that check out, AWQ W4 takes the biggest hit from INT2 KV. So the weight and KV settings should be picked together, not one at a time.
- **Memory:** key norm makes the KV cache 4.2× smaller (897 → 216 MiB at 7K tokens). Either mixed model with the key-norm cache comes to about 5.7 GB at 7K tokens, vs. 16.9 GB for FP16 everything.
- **Speed:** decoding is about 2.5× slower with a quantized cache (26–29 → 69–74 ms/token). There's no kernel yet, so the INT2 tokens are unpacked at every step.

## Limitations

- **The KV cache only runs in Hugging Face, not vLLM.** Aditya's `evaluate.py` runs models inside vLLM, which uses its own cache, so it can't measure the per-token part. His examples and scorers are reused here instead.
- **Weights are unpacked to FP16** so the KV cache can be swapped in. The weight GB column is the checkpoint size from Ushnesha's results, and decode time doesn't show any speedup from quantized weights. For that, see her vLLM numbers (decode time per token: 10.78 ms for FP16, 4.83 ms for mixed GPTQ).
- **No KV kernel**, so latency goes up instead of down. A fused kernel would be v2.
- **Two KV precisions only** (full precision and INT2), not the planned 2/4/8 bits.
- **Batch size 1**, so no throughput numbers.
- **HotpotQA uses 50 examples**, so small differences in F1 are noise.
- **The passkey filler is four sentences repeated**, which makes the passkey easier to spot than a fact in real text.
- **Exact match is lower than "found"** because many correct answers add words (e.g. "The secret passkey is: …"). The mixed GPTQ model does this most.