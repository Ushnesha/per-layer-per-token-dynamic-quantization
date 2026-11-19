"""Token-selection policies for the end-to-end tests (perplexity and passkey retrieval).

A policy gives every token an importance score. The tokens with the highest scores stay in
full precision and the rest are fake-quantized to a low bit-width. The attention sinks and
the most recent `window` tokens always stay in full precision.

How a perplexity run works, per batch of chunks:
    1. One full-precision forward pass computes each policy's scores.
    2. `build_bit_map` turns the scores into a bit map: sinks full precision, the top
       `budget` fraction of the tokens that will leave the window full precision,
       everything else `low_bits`.
    3. The chunk is streamed through a `FakeQuantKVCache` with that bit map, `stride`
       tokens at a time. A token stays in full precision while it is inside the recent
       window and is quantized (or not, per the bit map) once it leaves.

A passkey run (`policy_passkey`) does steps 1 and 2 on the prompt, processes the whole
prompt in full precision, quantizes the cache per the bit map, and then generates the
answer from the quantized cache, the way KIVI-style systems handle prefill.

Every score below only uses information that exists when a token leaves the window: the
token's own keys, its position, or the attention it received while inside the window. Two
simplifications apply equally to every policy: the scores come from a full-precision pass,
and the budget is ranked over the whole sequence, not only over the tokens seen so far.

Example:
    model = AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
    ppl = policy_perplexity(
        model, chunks, key_norm_linear_scores("results/key_norm_linear_smollm360m.pt"),
        budget=0.1, low_bits=2, group_size=32,
    )
"""

from __future__ import annotations

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
    """The linear model on per-layer key norms (+ position), loaded from its saved checkpoint.

    The features are built exactly as in training: the head-averaged key L2 norm of every
    layer, then position / num_positions if the checkpoint used position, then standardized
    with the saved mean and std. A checkpoint saved with "use_position": False is the
    key-norms-only model, needed for prompts longer than the training chunks. Checkpoints
    without that key used position, like the SmolLM one.
    """
    ckpt = torch.load(checkpoint_path)
    linear = nn.Linear(ckpt["mean"].shape[0], 1)
    linear.load_state_dict(ckpt["state_dict"])
    linear.eval()
    mean, std = ckpt["mean"], ckpt["std"]
    use_position = ckpt.get("use_position", True)

    @torch.no_grad()
    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        if model.name_or_path != ckpt["llm"]:
            warnings.warn(f"Checkpoint was trained on {ckpt['llm']!r}, but the model is {model.name_or_path!r}")

        outputs = model(input_ids=input_ids, use_cache=True)
        norms = key_l2_norm(outputs.past_key_values).cpu()  # [num_layers, batch, seq_len]
        if norms.shape[0] + int(use_position) != mean.shape[0]:
            raise ValueError(f"Checkpoint expects {mean.shape[0] - int(use_position)} layers, the model has {norms.shape[0]}")

        features = norms.permute(1, 2, 0)  # [batch, seq_len, num_layers]
        if use_position:
            batch, seq_len, _ = features.shape
            position = (torch.arange(seq_len) / ckpt["num_positions"]).expand(batch, seq_len)
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

    The attention weights are read one layer at a time through hooks and thrown away right
    after, so long prompts (7K tokens on Llama) fit in memory. Needs the model loaded with
    attn_implementation="eager", and a Llama-style model (`model.model.layers[i].self_attn`).
    """

    @torch.no_grad()
    def score(model: PreTrainedModel, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        idx = torch.arange(seq_len, device=input_ids.device)
        query, key = idx[:, None], idx[None, :]
        counted = query >= key  # [query, key]
        if window is not None:
            counted &= query < key + window

        per_layer = []

        def read_weights(module, args, output):
            weights = output[1]  # [batch, heads, query, key]
            if weights is None:
                raise ValueError("No attention weights were returned. Load the model with attn_implementation='eager'.")
            per_layer.append(weights.masked_fill(~counted, 0).sum(dim=-2, dtype=torch.float32).mean(dim=1))

        handles = [layer.self_attn.register_forward_hook(read_weights) for layer in model.model.layers]
        try:
            model(input_ids=input_ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        return torch.stack(per_layer).mean(dim=0).cpu()  # [batch, seq_len]

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


@torch.no_grad()
def generate_from_quantized_cache(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    bits: torch.Tensor,
    window: int,
    group_size: int | None = None,
    max_new_tokens: int = 32,
    stop_token_ids: tuple[int, ...] = (),
    stop_text: str | None = None,
    tokenizer=None,
) -> list[int]:
    """Greedy generation for one prompt, with the prompt processed in full precision first.

    1. Every prompt token except the last goes through the model with nothing quantized:
       the cache's window is temporarily larger than the prompt.
    2. The window shrinks back to `window`. On the next step (the last prompt token), every
       prompt token older than the window is quantized per `bits`, and from then on the
       answer is generated from the quantized cache.

    Args:
        model: A causal LM.
        input_ids: [1, prompt_len].
        bits: Bit map [1, prompt_len] from `build_bit_map`. Generated tokens are never quantized.
        window: Most recent tokens kept in full precision.
        group_size: Quantization group size along head_dim.
        max_new_tokens: Most tokens to generate.
        stop_token_ids: Stop as soon as one of these is generated (it isn't returned).
        stop_text: Stop after a token whose text contains this (e.g. "\\n"). Needs `tokenizer`.
        tokenizer: Used only to check `stop_text`.

    Returns:
        The generated token IDs.
    """
    if input_ids.shape[0] != 1:
        raise ValueError("generate_from_quantized_cache handles one prompt at a time")
    prompt_len = input_ids.shape[1]
    cache = FakeQuantKVCache(bits, window=prompt_len, group_size=group_size, config=model.config)
    if prompt_len > 1:
        model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)
    cache.window = window

    next_input = input_ids[:, -1:]
    new_tokens = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=next_input, past_key_values=cache, use_cache=True).logits[:, -1]
        token = int(logits.argmax(dim=-1))
        if token in stop_token_ids:
            break
        new_tokens.append(token)
        if stop_text is not None and stop_text in tokenizer.decode([token]):
            break
        next_input = torch.tensor([[token]], device=input_ids.device)
    return new_tokens


@torch.no_grad()
def policy_passkey(
    model: PreTrainedModel,
    tokenizer,
    examples: list[dict],
    score_fn: ScoreFn | None,
    budget: float,
    low_bits: int,
    is_correct: Callable[[str, str], bool],
    stop_token_ids: tuple[int, ...] = (),
    stop_text: str | None = "\n",
    group_size: int | None = None,
    window: int = 128,
    num_sinks: int = 4,
    max_new_tokens: int = 32,
    show_progress: bool = False,
) -> list[dict]:
    """Passkey retrieval with a policy deciding which prompt tokens stay in full precision.

    Built to take the examples from Aditya's `retrieval.build_example` and score them with
    his `retrieval.exact_match`, with the same greedy decoding (32 tokens, stop at a stop
    token or a newline). The difference is that generation runs in Hugging Face with the
    fake-quantized cache instead of vLLM.

    Baselines: score_fn=None with budget=1.0 is full precision, and with budget=0.0 uniform
    low-bit (sinks and window still full precision).

    Args:
        model: A causal LM. Eager attention if the policy needs attention weights.
        tokenizer: The model's tokenizer, for decoding the answer.
        examples: Dicts with at least "prompt_token_ids" (list of ints) and "expected".
        score_fn: One of the policies above, or None for the baselines.
        budget: Fraction of the competing prompt tokens kept in full precision.
        low_bits: Bit-width for the other prompt tokens.
        is_correct: (response, expected) -> bool, e.g. `exact_match`.
        stop_token_ids: Token IDs that end the answer, e.g. Aditya's `stop_token_ids(tokenizer)`.
        stop_text: The answer ends after a token containing this text. "\\n" matches vLLM's stop=['\\n'].
        group_size, window, num_sinks: As in `policy_perplexity`.
        max_new_tokens: Most tokens in an answer.
        show_progress: Show a progress bar.

    Returns:
        One dict per example: its fields (without the token IDs), plus "response" and "correct".
    """
    stop_token_ids = tuple(stop_token_ids)
    items = examples
    if show_progress:
        from tqdm.auto import tqdm

        items = tqdm(examples, desc="policy passkey")

    trials = []
    for example in items:
        input_ids = torch.tensor([example["prompt_token_ids"]], device=model.device)
        scores = score_fn(model, input_ids) if score_fn is not None else torch.zeros(input_ids.shape)
        bits = build_bit_map(scores, budget, low_bits, num_sinks, window)
        new_tokens = generate_from_quantized_cache(
            model, input_ids, bits, window, group_size, max_new_tokens, stop_token_ids, stop_text, tokenizer
        )
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if stop_text is not None:
            response = response.split(stop_text)[0]  # vLLM leaves the stop text out of the answer
        trials.append(
            {
                **{key: value for key, value in example.items() if key != "prompt_token_ids"},
                "response": response,
                "correct": is_correct(response, example["expected"]),
            }
        )
    return trials