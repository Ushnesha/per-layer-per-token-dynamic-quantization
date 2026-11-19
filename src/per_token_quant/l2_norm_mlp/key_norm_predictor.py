import torch
import torch.nn as nn
from scipy.stats import kendalltau
import os

from .mlp import ShallowMLP, train
from oracle.analysis_utils import sensitivity_scores, per_chunk_correlation, per_chunk_top_k_overlap


# Getting the data
ORACLE_PATH = "./oracle/results/oracle_kl_llama3_8b_instruct_int4_gfull_20x512.pt"
SIGNALS_PATH = "./oracle/results/signals_llama3_8b_instruct_20x512.pt"
USE_POSITION = False
OUTPUT_PATH = f"./l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct{'' if USE_POSITION else '_nopos'}.pt"
H = 64
SEED = 0

torch.manual_seed(SEED)

oracle = torch.load(ORACLE_PATH)
signals = torch.load(SIGNALS_PATH)

scores_h64 = sensitivity_scores(oracle["kl"], H)
key_norm = signals["key_norm"]

num_positions = scores_h64.shape[1]
key_norm = key_norm.permute(1, 2, 0)[:, :num_positions, :]

num_chunks = signals["num_chunks"]
if USE_POSITION:
    position_feature = torch.arange(num_positions).repeat(num_chunks, 1) / num_positions
    features = torch.cat((key_norm, position_feature.unsqueeze(-1)), dim=-1)
else:
    features = key_norm

# Splitting data
training_set = features[:10, :, :]
test_set = features[10:, :, :]

training_set = training_set.flatten(0, 1)

training_targets = torch.log(scores_h64[:10, :].clamp_min(1e-12)).flatten(0, 1)
testing_targets = torch.log(scores_h64[10:, :].clamp_min(1e-12))

# Standarizing
mean = training_set.mean(dim=0)
std = training_set.std(dim=0)

training_set = (training_set - mean) / std
test_set = (test_set - mean) / std

target_mean = training_targets.mean(dim=0)
target_std = training_targets.std(dim=0)

training_targets = (training_targets - target_mean) / target_std
testing_targets = (testing_targets - target_mean) / target_std

# Training
linear_model = nn.Linear(features.shape[-1], 1)
mlp = ShallowMLP(features.shape[-1], 8, 1)

linear_model_optim = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
mlp_optim = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
loss_fn = nn.MSELoss()
epochs = 2000

print("Training linear_model...")
train(linear_model, training_set, training_targets, linear_model_optim, loss_fn, epochs)

print("\nTraining mlp...")
train(mlp, training_set, training_targets, mlp_optim, loss_fn, epochs)

# Evaluation
linear_model.eval()
mlp.eval()
with torch.no_grad():
    linear_model_pred = linear_model(test_set).squeeze(-1)
    mlp_pred = mlp(test_set).squeeze(-1)

raw_attention = signals["raw_attention"].mean(dim=0)[10:]
plain_key_norm = -signals["key_norm"].mean(dim=0)[10:]

test_scores = scores_h64[10:]
test_entries = {
    "linear model": linear_model_pred,
    "mlp": mlp_pred,
    "raw attention": raw_attention,
    "plain key norm": plain_key_norm,
}

print("\nTest chunks (10-19):")
for name, pred in test_entries.items():
    tau = per_chunk_correlation(test_scores, pred, kendalltau).mean()
    top10 = per_chunk_top_k_overlap(test_scores, pred).mean()
    print(f"{name:15s} tau {tau:.3f} | top-10% {top10:.1%}")

# Overfitting check
train_set_by_chunk = (features[:10] - mean) / std
with torch.no_grad():
    linear_train_pred = linear_model(train_set_by_chunk).squeeze(-1)
    mlp_train_pred = mlp(train_set_by_chunk).squeeze(-1)

train_scores = scores_h64[:10]
print("\nTraining chunks (0-9):")
for name, pred in {"linear model": linear_train_pred, "mlp": mlp_train_pred}.items():
    tau = per_chunk_correlation(train_scores, pred, kendalltau).mean()
    print(f"{name:15s} tau {tau:.3f}")


# After reviewing the results, it seems like the linear model is better
# Saving linear model's weights
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
torch.save(
    {
        "state_dict": linear_model.state_dict(),
        "mean": mean,
        "std": std,
        "H": H,
        "llm": signals["model"],
        "training_chunks": list(range(10)),
        "seed": SEED,
        "num_positions": num_positions,
        "use_position": USE_POSITION,
    },
    OUTPUT_PATH,
)

# Sanity check
linear_model_experiment = torch.load(OUTPUT_PATH)
state_dict = linear_model_experiment["state_dict"]
weights = state_dict["weight"]
mean = linear_model_experiment["mean"]
std = linear_model_experiment["std"]

print(f"weights shape: {weights.shape}")
print(f"mean shape: {mean.shape}")
print(f"std shape: {std.shape}")