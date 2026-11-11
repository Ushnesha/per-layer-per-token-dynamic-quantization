# Per-token adaptive KV-cache quantization

The idea: keep the most important tokens in the KV cache at full precision and store the rest in fewer bits. This folder holds the code for that, starting with the **oracle study**. The oracle study measures how sensitive each token really is when it's quantized to INT4, then checks which cheap signal predicts that best.

## Files

**Helpers**

| File | What it does |
| --- | --- |
| `quant.py` | Fake quantization (round to N bits, convert back) and `FakeQuantKVCache`, a KV cache that quantizes each token according to a bit map |
| `signals.py` | `compute_signals`: raw accumulated attention (H2O), normalized attention (ZipCache) and key L2 norm (Devoto et al.), from one forward pass |
| `metrics.py` | `perplexity`: perplexity over chunks, either in one pass or streamed a few tokens at a time through a cache |
| `models.py` | Loads the model and tokenizer |
| `data.py` | `load_wikitext`: WikiText-2 (raw) test set, tokenized and cut into fixed-length chunks |
| `plots.py` | Plots for sensitivity by position, by window size, and signal quality by layer |
| `analysis_utils.py` | Sensitivity scores and grading: per-chunk Kendall τ and Spearman, top-10% overlap, summaries |
| `test_kvq.py` | Sanity tests on a tiny random model (no download, runs on CPU in seconds) |

**Oracle study** (run in this order)

| File | What it does | Output |
| --- | --- | --- |
| `oracle.py` | Quantizes one token at a time to INT4 and measures the KL divergence at every later position. Needs a GPU: 512 forward passes | `results/oracle_kl_smollm360m_int4_gfull_20x512.pt` |
| `oracle_analysis.py` | Turns the KL grid into one sensitivity score per token, checks that the window size H doesn't change the ranking, and looks at the biggest spike | `results/sensitivity_H64.png`, `results/sensitivity_by_H.png` |
| `signals_run.py` | Computes the three signals on the same chunks. Needs a GPU | `results/signals_smollm360m_20x512.pt` |
| `signals_analysis.py` | Grades each signal against the oracle, plus position and random baselines, and grades each layer on its own | `results/tau_by_layer.png` |

## Setup

- Python 3.10+
- `torch`, `transformers` (tested with 5.17.0), `datasets`, `scipy`, `numpy`, `matplotlib`
- Model: `HuggingFaceTB/SmolLM-360M`. It's open, so no Hugging Face login is needed.

## How to run

Run everything **from inside this folder**. The scripts save to `results/` relative to where you run them, and running from the repo root would put the files in the team's main `results/` folder.

```bash
cd src/per_token_quant/oracle_study

python test_kvq.py           # all 10 tests should pass
python oracle.py             # GPU, the slow step
python oracle_analysis.py
python signals_run.py        # GPU
python signals_analysis.py
```


## Main results (SmolLM-360M, 20 chunks of 512 tokens, INT4)

- The first token is about 15× more sensitive than a typical token (an attention sink). After the first ~50 tokens, position says little about sensitivity.
- The window size H (32, 64 or 128) barely changes the ranking: Kendall τ is 0.86–0.93 between them.
- Raw attention matches the oracle best (Kendall τ = 0.35, vs. 0.11 for position). Normalized attention scores 0.24, and key norm only 0.07, even when it skips layers 0 and 1 like its paper does.

## Notes

- Everything runs in FP32, so the oracle's tiny KL values aren't buried in rounding noise.
- The signals need the model loaded with eager attention, because that's the only mode that returns attention weights.
- The full write-up is in the oracle study report.