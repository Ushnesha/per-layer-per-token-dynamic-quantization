import torch
from scipy.stats import kendalltau, spearmanr
from transformers import AutoTokenizer

from data import load_wikitext
from plots import plot_sensitivity, plot_sensitivity_by_window
from analysis_utils import sensitivity_scores, shared_positions, per_chunk_correlation, per_chunk_top_k_overlap, compare_windows, print_summary, inspect_token

ORACLE_PATH = "results/oracle_kl_smollm360m_int4_gfull_20x512.pt"
MODEL_NAME = "HuggingFaceTB/SmolLM-360M"
NUM_CHUNKS = 20
CHUNK_SIZE = 512
WINDOWS = [32, 64, 128]
WINDOW_PAIRS = [(32, 64), (32, 128), (64, 128)]
SPIKE_POSITION = 343

# Load the oracle results and the same chunks the oracle used
kl_res = torch.load(ORACLE_PATH)["kl"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
chunks = load_wikitext(tokenizer=tokenizer, chunk_size=CHUNK_SIZE)[:NUM_CHUNKS]

# One sensitivity score per token, for each window size
scores = {H: sensitivity_scores(kl_res, H) for H in WINDOWS}

plot_sensitivity(scores[64], window=64, path="results/sensitivity_H64.png")
plot_sensitivity_by_window(scores, path="results/sensitivity_by_H.png")

# Does the window size change the ranking, or the top 10% of tokens?
kendall_stats = compare_windows(scores, WINDOW_PAIRS, lambda a, b: per_chunk_correlation(a, b, kendalltau))
spearman_stats = compare_windows(scores, WINDOW_PAIRS, lambda a, b: per_chunk_correlation(a, b, spearmanr))
overlap_stats = compare_windows(scores, WINDOW_PAIRS, per_chunk_top_k_overlap)
print_summary(kendall_stats, "tau")
print_summary(spearman_stats, "spearman")
print_summary(overlap_stats, "top-10% overlap")

# The spike at token 343: find the chunk causing it and look at the token
spike_chunk = torch.argmax(scores[64][:, SPIKE_POSITION]).item()
inspect_token(kl_res, chunks, tokenizer, spike_chunk, SPIKE_POSITION)