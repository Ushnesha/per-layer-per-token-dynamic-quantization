"""Token-selection policies for the end-to-end perplexity test.

A policy gives every token an importance score. The tokens with the highest scores stay in
FP16 and the rest are fake-quantized to a low bit-width. The attention sinks and the most
recent `window` tokens always stay in FP16.

How a run works, per batch of chunks:
    1. One full-precision forward pass computes each policy's scores.
    2. `build_bit_map` turns the scores into a bit map: sinks FP16, the top `budget`
       fraction of the tokens that will leave the window FP16, everything else `low_bits`.
    3. The chunk is streamed through a `FakeQuantKVCache` with that bit map, `stride`
       tokens at a time. A token stays in FP16 while it is inside the recent window and is
       quantized (or not, per the bit map) once it leaves.

Every score below only uses information that exists when a token leaves the window: the
token's own keys, its position, or the attention it received while inside the window. Two
simplifications apply equally to every policy: the scores come from a full-precision pass,
and the budget is ranked over the whole chunk, not only over the tokens seen so far.

Example:
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
    ppl = policy_perplexity(
        model, chunks, key_norm_linear_scores("results/key_norm_linear_smollm360m.pt"),
        budget=0.1, low_bits=2, group_size=32,
    )
"""

import math
import warnings
from typing import Callable

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from oracle.metrics import _streamed_nll
from oracle.quant import FULL_PRECISION, FakeQuantKVCache
from oracle.signals import key_l2_norm

# (model, input_ids [batch, seq_len]) -> scores [batch, seq_len]; higher means keep in FP16
ScoreFn = Callable[[PreTrainedModel, torch.Tensor], torch.Tensor]


def build_bit_map(
    scores: torch.Tensor, budget: float, low_bits: int, num_sinks: int = 4, window: int = 0
) -> torch.Tensor:
    """Turns importance scores into a bit-width per token.

    Only the tokens that will actually leave the window compete for the budget: positions
    num_sinks .. seq_len - window - 1. The last `window` tokens never leave it, so they stay
    in FP16 anyway, and a policy that picked them would waste part of its budget.

    Args:
        scores: [batch, seq_len], higher means more important.
        budget: Fraction of the competing tokens kept in FP16, from 0.0 to 1.0.
        low_bits: Bit-width for the competing tokens that don't make the cut.
        num_sinks: The first tokens of each sequence, always kept in FP16.
        window: The cache's recent window. 0 means every non-sink token competes.

    Returns:
        Long tensor [batch, seq_len] for `FakeQuantKVCache`, with 16 meaning FP16.
    """
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be between 0 and 1, got {budget}")
    scores = scores.float().cpu()
    batch, seq_len = scores.shape
    end = seq_len - window

    bits = torch.full((batch, seq_len), FULL_PRECISION, dtype=torch.long)
    bits[:, num_sinks:end] = low_bits

    competing = scores[:, num_sinks:end]
    k = int(round(budget * competing.shape[1]))
    if k > 0:
        top = competing.topk(k, dim=-1).indices + num_sinks  # back to positions in the full sequence
        bits.scatter_(1, top, FULL_PRECISION)
    return bits


# ---------------------------------------------------------------- policies


def key_norm_linear_scores(checkpoint_path: str) -> ScoreFn:
    """The linear model on per-layer key norms + position, loaded from its saved checkpoint.

    The features are built exactly as in training: the head-averaged key L2 norm of every
    layer, then position / num_positions, then standardized with the saved mean and std.
    """
    ckpt = torch.load(checkpoint_path)
    linear = nn.Linear(ckpt["mean"].shape[0], 1)
    linear.load_state_dict(ckpt["state_dict"])
    linear.eval()
    mean, std, num_positions = ckpt["mean"], ckpt["std"], ckpt["num_positions"]

    @torch.no_grad()
    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        if model.name_or_path != ckpt["llm"]:
            warnings.warn(f"Checkpoint was trained on {ckpt['llm']!r}, but the model is {model.name_or_path!r}")

        outputs = model(input_ids=input_ids, use_cache=True)
        norms = key_l2_norm(outputs.past_key_values).cpu()  # [num_layers, batch, seq_len]
        if norms.shape[0] + 1 != mean.shape[0]:
            raise ValueError(f"Checkpoint expects {mean.shape[0] - 1} layers, the model has {norms.shape[0]}")

        features = norms.permute(1, 2, 0)  # [batch, seq_len, num_layers]
        batch, seq_len, _ = features.shape
        position = (torch.arange(seq_len) / num_positions).expand(batch, seq_len)
        features = torch.cat((features, position.unsqueeze(-1)), dim=-1)
        features = (features - mean) / std
        return linear(features).squeeze(-1)  # predicted sensitivity: higher = keep

    return score


def windowed_attention_scores(window: int | None = 128) -> ScoreFn:
    """H2O-style accumulated attention, counting only the queries a token sees while in the window.

    Token j's score is the attention it receives from queries j .. j + window - 1, averaged
    over heads and layers. That is what an H2O-style policy has accumulated when the token
    leaves a window of that size. Use the same window as the cache.

    window=None sums over every later query in the chunk, like the oracle study's raw
    attention. That peeks at queries that come after the decision, so it flatters attention.

    Needs the model loaded with attn_implementation="eager".
    """

    @torch.no_grad()
    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids, use_cache=False, output_attentions=True)
        if outputs.attentions is None or any(a is None for a in outputs.attentions):
            raise ValueError("The model did not return attention weights. Load it with attn_implementation='eager'.")

        seq_len = input_ids.shape[1]
        idx = torch.arange(seq_len, device=outputs.attentions[0].device)
        query, key = idx[:, None], idx[None, :]
        counted = query >= key  # [query, key]
        if window is not None:
            counted &= query < key + window

        total = 0.0
        for attn in outputs.attentions:  # [batch, heads, query, key]
            total = total + (attn.float() * counted).sum(dim=-2).mean(dim=1)  # [batch, key]
        return (total / len(outputs.attentions)).cpu()

    return score


def position_scores() -> ScoreFn:
    """Earlier tokens first, the same as the oracle study's position baseline."""

    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_ids.shape
        return -torch.arange(seq_len, dtype=torch.float32).expand(batch, seq_len)

    return score


def random_scores(seed: int = 0) -> ScoreFn:
    """Random scores. Make a new one for each run so every run sees the same random maps."""
    generator = torch.Generator().manual_seed(seed)

    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.rand(input_ids.shape, generator=generator)

    return score


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def policy_perplexity(
    model: PreTrainedModel,
    chunks: torch.Tensor,
    score_fn: ScoreFn | None,
    budget: float,
    low_bits: int,
    group_size: int | None = None,
    window: int = 128,
    num_sinks: int = 4,
    stride: int = 64,
    batch_size: int = 1,
    show_progress: bool = False,
) -> float:
    """Streamed perplexity with a policy deciding which tokens stay in FP16.

    Baselines: score_fn=None with budget=0.0 is uniform low-bit (sinks and the window
    still FP16), and score_fn=None with budget=1.0 is all FP16.

    Args:
        model: A causal LM. Eager attention if any policy needs attention weights.
        chunks: Token IDs [num_chunks, chunk_len].
        score_fn: One of the policies above, or None for the baselines.
        budget: Fraction of the tokens that leave the window (sinks excluded) kept in FP16.
        low_bits: Bit-width for the other tokens.
        group_size: Quantization group size along head_dim. None uses the whole head_dim.
        window: Most recent tokens kept in FP16.
        num_sinks: First tokens always kept in FP16.
        stride: Tokens per forward pass while streaming. Keep it at or below `window`.
        batch_size: Chunks per forward pass.
        show_progress: Show a progress bar.

    Returns:
        The perplexity as a float.
    """
    starts = range(0, chunks.shape[0], batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        starts = tqdm(starts, desc="policy perplexity")

    total_nll = 0.0
    total_tokens = 0
    for i in starts:
        batch = chunks[i : i + batch_size].to(model.device)
        scores = score_fn(model, batch) if score_fn is not None else torch.zeros(batch.shape)
        bits = build_bit_map(scores, budget, low_bits, num_sinks, window)

        def make_cache():
            return FakeQuantKVCache(bits, window=window, group_size=group_size, config=model.config)

        total_nll += _streamed_nll(model, batch, make_cache, stride)
        total_tokens += batch.shape[0] * (batch.shape[1] - 1)

    return math.exp(total_nll / total_tokens)