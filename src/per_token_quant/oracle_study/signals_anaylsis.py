import torch
from scipy.stats import kendalltau, spearmanr

from analysis_utils import sensitivity_scores, per_chunk_correlation, per_chunk_top_k_overlap, print_summary
from plots import plot_tau_by_layer

ORACLE_PATH = "results/oracle_kl_smollm360m_int4_gfull_20x512.pt"
SIGNALS_PATH = "results/signals_smollm360m_20x512.pt"
SEED = 0

# Loading both files
kl = torch.load(ORACLE_PATH)["kl"]
scores_h64 = sensitivity_scores(kl, 64)

signals = torch.load(SIGNALS_PATH)
chunk_size = signals["chunk_size"]
num_chunks = signals["num_chunks"]

# Average over layers
signal_averages = {signal_name: signals[signal_name].mean(dim=0) for signal_name in ["raw_attention", "normalized_attention", "key_norm"]}

# Flipping key norm
signal_averages["negated_key_norm"] = -signal_averages.pop("key_norm")

# Adding two baselines
torch.manual_seed(SEED)
positions = -torch.arange(chunk_size).repeat(num_chunks, 1)
random_scores = torch.rand(num_chunks, chunk_size)

# Grouping everything into a dictionary
entries = {
    **signal_averages,
    "positions": positions,
    "random_scores": random_scores
}

# Key norm the paper's way: skip layers 0 and 1
entries["negated_key_norm_skip01"] = -signals["key_norm"][2:].mean(dim=0)

# Grading each entry against the oracle
kendalltau_comparison = {entry_name: per_chunk_correlation(scores_h64, entry, kendalltau) for entry_name, entry in entries.items()}
spearman_comparison = {entry_name: per_chunk_correlation(scores_h64, entry, spearmanr) for entry_name, entry in entries.items()}
top_10_comparison = {entry_name: per_chunk_top_k_overlap(scores_h64, entry) for entry_name, entry in entries.items()}

# without the sink
kendalltau_no_sink = {entry_name: per_chunk_correlation(scores_h64[:, 1:], entry[:, 1:], kendalltau) for entry_name, entry in entries.items()}
spearman_no_sink = {entry_name: per_chunk_correlation(scores_h64[:, 1:], entry[:, 1:], spearmanr) for entry_name, entry in entries.items()}
top_10_no_sink = {entry_name: per_chunk_top_k_overlap(scores_h64[:, 1:], entry[:, 1:]) for entry_name, entry in entries.items()}

# Printing results
print_summary(kendalltau_comparison, "Kendall Tau Comparison")
print_summary(spearman_comparison, "Spearman Comparison")
print_summary(top_10_comparison, "Top-10% Comparison")
print("\n")
print_summary(kendalltau_no_sink, "Kendall Tau No Sink Comparison")
print_summary(spearman_no_sink, "Spearman No Sink Comparison")
print_summary(top_10_no_sink, "Top-10% No Sink Comparison")

# Grading each layer on its own
num_layers = signals["raw_attention"].shape[0]
signs = {"raw_attention": 1, "normalized_attention": 1, "key_norm": -1}
tau_layer_results = {
    name: torch.stack([per_chunk_correlation(scores_h64, sign * signals[name][layer], kendalltau).mean()
                       for layer in range(num_layers)])
    for name, sign in signs.items()
}

# Plotting tau by layer
plot_tau_by_layer(
    tau_layer_results,
    baselines={"position": 0.113, "random": 0.003},
    path="results/tau_by_layer.png",
)
