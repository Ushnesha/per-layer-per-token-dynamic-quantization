# Per-token adaptive KV-cache quantization

Keep the most important tokens in the KV cache at full precision and store the rest in fewer bits. This folder holds the whole line of work, in three steps:

1. **`oracle/`**: the oracle study. It measures how sensitive each token really is to quantization, and checks which cheap signal predicts that best.
2. **`l2_norm_mlp/`**: a small model that predicts token importance from key norms, plus token-selection policies tested end to end on perplexity.
3. **`prototype/`**: prototype v0, a working system. Text goes in, token importance comes from key norms, the KV cache is quantized, and text comes out.

Everything uses **fake quantization** for now: values are rounded to low bits and stored back at full precision. That measures accuracy correctly but doesn't save memory yet. Real low-bit storage is v1.

## How to run

Run everything **from this folder** (`src/per_token_quant`), as modules:

```bash
python -m oracle.test_kvq                 # sanity tests, tiny CPU model
python -m l2_norm_mlp.test_policies
python -m prototype.test_adaptive_kv
```

Setup: Python 3.10+, `torch`, `transformers` (5.x), `datasets`, `scipy`, `numpy`, `matplotlib`, `tqdm`. Llama-3 is a gated model, so log in once with `huggingface-cli login`. Never put the token in code.

## Files

**`oracle/`**: the oracle study

| File | What it does |
| --- | --- |
| `quant.py` | Fake quantization, and `FakeQuantKVCache`, a cache that quantizes each token per a bit map |
| `signals.py` | Raw attention (H2O), normalized attention (ZipCache) and key L2 norm (Devoto et al.) from one forward pass |
| `metrics.py` | Perplexity, in one pass or streamed a few tokens at a time |
| `models.py`, `data.py` | Load the model and tokenizer; WikiText-2 test set cut into fixed-length chunks |
| `analysis_utils.py` | Sensitivity scores and grading (Kendall τ, Spearman, top-10% overlap) |
| `plots.py` | Sensitivity and per-layer τ plots |
| `oracle.py` | SmolLM-360M oracle: quantizes one token at a time to INT4 and measures the KL divergence at every later position (GPU) |
| `llama_oracle.py` | The same oracle plus the signals for Llama-3-8B-Instruct (A100 80GB, about 75 minutes) |
| `oracle_analysis.py` | Turns the KL grid into one score per token, checks the window size, looks at the biggest spike |
| `signals_run.py`, `signals_anaylsis.py` | Computes the signals (GPU) and grades them against the oracle, per layer too |
| `test_kvq.py` | Sanity tests |
| `Oracle_study_report.pdf` | Write-up of the SmolLM oracle study |

**`l2_norm_mlp/`**: predicting importance from key norms

| File | What it does |
| --- | --- |
| `mlp.py` | A small MLP and the training loop |
| `key_norm_predictor.py` | Trains a linear model (and the MLP) on per-layer key norms to predict the oracle's sensitivity. `USE_POSITION` adds position as a feature |
| `policies.py` | Token-selection policies (linear key-norm model, windowed attention, position, random) and `policy_perplexity` / `policy_passkey` to test them |
| `perplex_test.py` | SmolLM-360M perplexity with each policy at 10% and 20% budgets |
| `test_policies.py` | Sanity tests |

**`prototype/`**: prototype v0

| File | What it does |
| --- | --- |
| `adaptive_kv.py` | `AdaptiveKVCache` scores tokens from the key norms it already stores, keeps the top fraction at full precision and quantizes the rest as they leave the recent window. `generate()` takes text in and gives text out. `memory_report()` gives the memory a real low-bit cache would use |
| `try_v0.py` | Smoke test on Llama-3-8B-Instruct with a long prompt |
| `passkey_v0.py` | Passkey retrieval through v0, using the same examples and scoring as `src/evaluation/benchmark/retrieval.py` |
| `test_adaptive_kv.py` | Sanity tests |
| `packing.py` | Real low-bit storage: values to 2-bit codes packed four per byte, with an FP16 scale and zero point per group, and back |
| `real_kv.py` | Prototype v1: `RealQuantKVCache`, the same decisions as v0, with low-bit tokens really stored packed and each layer rebuilt only while its attention runs |
| `test_packing.py`, `test_real_kv.py` | Sanity tests for v1 |
| `try_v1.py` | v1 smoke test: the same prompt as `try_v0`, printing stored, predicted and measured KV memory |

Results (`.pt` files, plots) are saved in each folder's `results/`.

## Results so far

**Oracle study, SmolLM-360M** (20 chunks of 512 tokens, INT4): how well each cheap signal ranks tokens like the oracle (Kendall τ)

| Signal | τ |
| --- | --- |
| Raw accumulated attention (H2O) | 0.351 |
| Normalized attention (ZipCache) | 0.235 |
| Position | 0.113 |
| Key norm, averaged over layers | 0.073 |
| Random | 0.003 |

Key norm flips sign across layers, so averaging the layers cancels it out.

**Learned key-norm model** (trained on chunks 0–9, tested on chunks 10–19)

| | SmolLM-360M τ / top-10% | Llama-3-8B-Instruct τ / top-10% |
| --- | --- | --- |
| Linear model on key norms (+ position on SmolLM) | 0.335 / 31.8% | 0.290 / 30.2% (no position) |
| Raw attention | 0.360 / 27.7% | 0.344 / 34.3% |
| Plain key norm | 0.078 / 14.3% | 0.166 / 28.0% |

Weighting the layers fixes the sign flip. On Llama, position adds nothing. A small MLP was no better than the linear model and overfit.

**Perplexity, SmolLM-360M** (WikiText-2, 512-token chunks after the first 20, INT2 group 32, 128-token full-precision window, 4 sinks): full precision 17.347, uniform INT2 18.478

| Policy | 10% kept | 20% kept |
| --- | --- | --- |
| Linear key-norm model | 18.277 | 18.118 |
| Windowed attention | 18.283 | 18.106 |
| Position | 18.263 | 18.085 |
| Random | 18.380 | 18.286 |

**Prototype v0, Llama-3-8B-Instruct** (BF16, A100 20 GB slice, INT2 group 32, 128-token window, 4 sinks, one ~2,070-token prompt)

| | Full-precision tokens | Average bits per KV value | KV compression vs. FP16 | Decode |
| --- | --- | --- | --- | --- |
| Full precision | all 2,168 | 16.0 | 1.00× | 47 ms/token |
| Key-norm policy, 10% kept | 333 | 5.0 | 3.20× | 66 ms/token |
| Uniform INT2 | 132 (sinks + window) | 3.8 | 4.22× | 61 ms/token |

Compression is what a real low-bit cache would use, counting the scales and zero points. v0 still stores everything at full precision, so GPU memory doesn't drop yet (that's v1). Decoding is slower because fake quantization adds work at every step.

**Passkey retrieval through v0, Llama-3-8B-Instruct** (Aditya's synthetic passkey examples: 2,048- and 7,168-token prompts, passkey at 10/50/90% depth, 10 examples per cell, 60 in total; scored by whether the answer contains the passkey)

Calibration with uniform quantization (only the sinks and the window kept): full precision 100%, INT2 group 32 100% (too mild to matter), INT2 group 128 80%, INT1 group 32 0%. The policies are compared at INT2 group 128:

| Policy, 10% kept | Passkey found | Passkey sentence kept in full precision |
| --- | --- | --- |
| Key-norm linear model | 100% | 97% |
| Random | 85% | 11% |
| Position | 80% | 0% |
| Uniform INT2 (nothing kept) | 80% | — |

The key-norm policy finds the passkey sentence on its own, without attention scores, and keeps retrieval at full-precision accuracy. Caveat: the filler is four sentences repeated, so the passkey sentence stands out more than a fact would in real text. Real documents (LongBench) are the next test. Also, many correct answers start with "The secret passkey is:", so the exact-match scorer in `retrieval.py` marks them wrong (78% exact match at full precision, vs. 100% found).

**LongBench HotpotQA through v0, Llama-3-8B-Instruct** (first 50 examples, 7,168-token prompts trimmed as in `long_context.py`, INT2 group 128, 10% kept, F1)

| Setting | F1 |
| --- | --- |
| Full precision | 40.8 |
| Uniform INT2 | 33.1 |
| Position, 10% | 35.7 |
| Key-norm linear model, 10% | 34.1 |
| Random, 10% | 34.0 |

On real multi-document QA, quantization costs 7.7 F1, and no policy recovers much of it at a 10% budget. With 50 examples, the 1–2 F1 differences between policies are within noise. Key norm does no better than random here. It flags tokens that look unusual, which works for the passkey (one odd sentence in repeated filler) but not when which passage matters depends on the question. Key norm never sees the question. A query-aware signal, such as attention from the question tokens, is the natural next step, at the cost of needing attention scores.

**Prototype v1, real low-bit storage, Llama-3-8B-Instruct** (the same prompt and settings as the v0 smoke test)

| Setting | KV cache stored | v0's prediction | Measured on GPU | Decode |
| --- | --- | --- | --- | --- |
| Full precision | 271.0 MiB | 271.0 MiB | 280.1 MiB | 43.9 ms/token |
| Key-norm policy, 10% | 84.6 MiB (3.2× smaller) | 84.6 MiB | 85.0 MiB | 87.9 ms/token |
| Uniform INT2 | 64.2 MiB (4.2× smaller) | 64.2 MiB | 64.5 MiB | 87.4 ms/token |

The stored size matches v0's prediction exactly, and the GPU's own memory counter agrees within half a MiB, so the savings are real. Decoding is about 2× slower because every layer is unpacked and rebuilt at every step. A kernel that reads the packed codes directly (v2) would remove that. The answers match v0's closely: v1 stores scales in FP16, v0 in FP32.