"""Prototype v0: adaptive per-token KV-cache quantization, end to end.

Input -> token salience -> quantized KV cache -> output, in one generation loop:

    1. The prompt is processed in full precision (prefill). While each layer writes its
       keys into the cache, the cache records every token's key L2 norm for that layer,
       so salience costs no extra forward pass and no attention scores.
    2. When a token leaves the recent window, the scorer turns its per-layer key norms
       into an importance score. Tokens scoring at least the threshold stay in full
       precision; the rest are quantized to `low_bits`. The attention sinks always stay in
       full precision.
    3. Generation continues from the mixed-precision cache. New tokens are scored and
       quantized the same way when they leave the window.

The threshold is set the first time non-sink tokens leave the window: it is the score that
keeps the top `budget` fraction of those tokens. Normally that is the whole prompt, at the
first decoding step. Later tokens are compared against the same threshold, so the budget
holds approximately. A prompt shorter than the window gives a noisy threshold.

The first generated token comes from the full-precision prefill, as in KIVI-style systems.
Quantization applies from the second generated token on.

v0 uses fake quantization: values are rounded to `low_bits` and stored back at the model's
precision. The outputs are what a real low-bit cache would produce, but no memory is saved
yet. `memory_report()` gives the memory a real implementation would use (that is v1).

Batch size 1 only.

Example:
    scorer = KeyNormLinearScorer("results/key_norm_linear_llama3_8b_instruct_nopos.pt")
    make_cache = lambda: AdaptiveKVCache(scorer, budget=0.1, low_bits=2, group_size=32, config=model.config)
    out = generate(model, tokenizer, "Summarize the following ...", make_cache, max_new_tokens=128)
    print(out["text"])
    print(out["cache"].memory_report())
"""

from __future__ import annotations

import math
import time
from typing import Callable

import torch
import torch.nn as nn
from transformers import DynamicCache, PreTrainedModel

from oracle.quant import FULL_PRECISION, fake_quantize_tokens, get_layer_kv

# ---------------------------------------------------------------- scorers
# A scorer takes key_norms [num_tokens, num_layers] and positions [num_tokens] and returns
# scores [num_tokens], where higher means more important.


class KeyNormLinearScorer:
    """The trained linear model: per-layer key norms (+ position, if it was trained with it) -> score."""

    def __init__(self, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path)
        self.linear = nn.Linear(ckpt["mean"].shape[0], 1)
        self.linear.load_state_dict(ckpt["state_dict"])
        self.linear.eval()
        self.mean, self.std = ckpt["mean"], ckpt["std"]
        self.use_position = ckpt.get("use_position", True)
        self.num_positions = ckpt.get("num_positions")
        self.llm = ckpt.get("llm")

    @torch.no_grad()
    def __call__(self, key_norms: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        features = key_norms
        if self.use_position:
            features = torch.cat((features, (positions / self.num_positions).unsqueeze(-1)), dim=-1)
        if features.shape[-1] != self.mean.shape[0]:
            raise ValueError(
                f"The checkpoint expects {self.mean.shape[0]} features, got {features.shape[-1]}. "
                f"Was it trained on this model ({self.llm})?"
            )
        return self.linear((features - self.mean) / self.std).squeeze(-1)


class PositionScorer:
    """Earlier tokens first, the oracle study's position baseline."""

    def __call__(self, key_norms: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return -positions.float()


class RandomScorer:
    """Random scores. Make a new one per run so every run sees the same choices."""

    def __init__(self, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, key_norms: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return torch.rand(positions.shape[0], generator=self.generator)


Scorer = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

# ---------------------------------------------------------------- cache


class AdaptiveKVCache(DynamicCache):
    """A KV cache that scores tokens from their key norms and quantizes the unimportant ones.

    Decisions happen at the start of each forward pass (in layer 0's update), before any
    attention runs: every token that will be outside the window once this pass's tokens are
    added is scored, and quantized in every layer if it falls below the threshold. During
    prefill nothing is old enough yet, so the prompt is processed in full precision.

    Args:
        scorer: Turns key norms (and positions) into importance scores.
        budget: Fraction of the tokens that leave the window (sinks excluded) kept in full precision.
        low_bits: Bit-width for the other tokens.
        group_size: Quantization group size along head_dim. None uses the whole head_dim.
        window: Most recent tokens kept in full precision.
        num_sinks: First tokens always kept in full precision.
        **kwargs: Passed to DynamicCache. `config=model.config` is required.

    After generation, `bits` holds the bit-width chosen for every decided token and
    `scores` their scores.
    """

    def __init__(
        self,
        scorer: Scorer,
        budget: float,
        low_bits: int,
        group_size: int | None = None,
        window: int = 128,
        num_sinks: int = 4,
        **kwargs,
    ):
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"budget must be between 0 and 1, got {budget}")
        if kwargs.get("config") is None:
            raise ValueError("Pass config=model.config")
        super().__init__(**kwargs)
        self.scorer = scorer
        self.budget = budget
        self.low_bits = low_bits
        self.group_size = group_size
        self.window = window
        self.num_sinks = num_sinks
        self.num_model_layers = kwargs["config"].num_hidden_layers

        self.key_norms: list[list[torch.Tensor]] = [[] for _ in range(self.num_model_layers)]
        self.num_tokens = 0  # tokens stored so far
        self.num_decided = 0  # tokens whose precision has been decided
        self.threshold: float | None = None
        self.bits = torch.empty(0, dtype=torch.long)
        self.scores = torch.empty(0)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if key_states.shape[0] != 1:
            raise ValueError("AdaptiveKVCache supports batch size 1")
        if layer_idx == 0:
            num_new = key_states.shape[-2]
            # Tokens outside the window once this pass's tokens are added. Only tokens from
            # earlier passes qualify, because they're the ones with key norms in every layer.
            self._decide_and_quantize(end=min(self.num_tokens + num_new - self.window, self.num_tokens))
            self.num_tokens += num_new

        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        # Head-averaged key L2 norm of the new tokens, the same feature the linear model was trained on
        self.key_norms[layer_idx].append(key_states.float().norm(dim=-1).mean(dim=1)[0])
        return keys, values

    def _decide_and_quantize(self, end: int) -> None:
        start = self.num_decided
        if end <= start:
            return

        positions = torch.arange(start, end)
        for layer in range(self.num_model_layers):
            self.key_norms[layer] = [torch.cat(self.key_norms[layer])]  # keep one tensor per layer
        norms = torch.stack([self.key_norms[layer][0][start:end] for layer in range(self.num_model_layers)], dim=-1)
        scores = self.scorer(norms.cpu(), positions).float().cpu()

        competing = positions >= self.num_sinks
        if self.threshold is None and competing.any():
            self.threshold = self._threshold_for(scores[competing])
        threshold = self.threshold if self.threshold is not None else math.inf
        bits = torch.full(scores.shape, self.low_bits, dtype=torch.long)
        bits[scores >= threshold] = FULL_PRECISION
        bits[~competing] = FULL_PRECISION

        if (bits < FULL_PRECISION).any():
            for layer in range(self.num_model_layers):
                keys, values = get_layer_kv(self, layer)
                token_bits = bits[None].to(keys.device)  # [1, num_decided_now]
                keys[:, :, start:end] = fake_quantize_tokens(keys[:, :, start:end], token_bits, self.group_size)
                values[:, :, start:end] = fake_quantize_tokens(values[:, :, start:end], token_bits, self.group_size)

        self.bits = torch.cat((self.bits, bits))
        self.scores = torch.cat((self.scores, scores))
        self.num_decided = end

    def _threshold_for(self, scores: torch.Tensor) -> float:
        """The score that keeps the top `budget` fraction of `scores`."""
        if self.budget >= 1.0:
            return -math.inf
        k = int(round(self.budget * scores.numel()))
        if k == 0:
            return math.inf
        return scores.topk(k).values[-1].item()

    def memory_report(self) -> dict:
        """KV memory a real implementation of this cache would use, vs. a plain FP16 cache.

        Full-precision tokens count as 16 bits per value, the way a deployment would store
        them, even if the model runs in FP32 here. Low-bit tokens count `low_bits` per value
        plus a 16-bit scale and a 16-bit zero point per quantization group. Tokens still in
        the window count as full precision.
        """
        keys, _ = get_layer_kv(self, 0)
        _, kv_heads, num_tokens, head_dim = keys.shape
        values_per_token = 2 * self.num_model_layers * kv_heads * head_dim  # keys + values, all layers
        group = head_dim if self.group_size is None else self.group_size
        groups_per_token = values_per_token // group

        num_low = int((self.bits < FULL_PRECISION).sum())
        num_full = num_tokens - num_low
        full_bytes = num_full * values_per_token * 2
        low_bytes = num_low * (values_per_token * self.low_bits / 8 + groups_per_token * 2 * 2)
        fp16_bytes = num_tokens * values_per_token * 2
        kv_bytes = full_bytes + low_bytes
        return {
            "tokens": num_tokens,
            "full_precision_tokens": num_full,
            "low_bit_tokens": num_low,
            "kv_bytes": kv_bytes,
            "fp16_kv_bytes": fp16_bytes,
            "compression": fp16_bytes / kv_bytes,
            "average_bits_per_value": 8 * kv_bytes / (num_tokens * values_per_token),
        }


# ---------------------------------------------------------------- generation


def _sync(model: PreTrainedModel) -> None:
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer,
    prompt: str | list[int],
    make_cache: Callable[[], AdaptiveKVCache],
    max_new_tokens: int = 128,
    stop_token_ids: list[int] | None = None,
    stop_text: str | None = None,
) -> dict:
    """Greedy generation with an adaptive KV cache: text (or token IDs) in, text out.

    Args:
        model: A causal LM. Any attention implementation works, since salience comes from
            key norms, not attention weights.
        tokenizer: The model's tokenizer.
        prompt: Text, or token IDs that are already formatted (e.g. with a chat template).
        make_cache: Returns a fresh AdaptiveKVCache.
        max_new_tokens: Most tokens to generate.
        stop_token_ids: Generation ends at any of these. None uses the model's EOS token(s).
        stop_text: Generation ends after a token whose text contains this (e.g. "\\n"), and
            the answer is cut before it.

    Returns:
        A dict with "text", "token_ids", "cache" (for `memory_report()`, `bits`, `scores`),
        "prefill_seconds" and "decode_seconds_per_token".
    """
    if isinstance(prompt, str):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    else:
        input_ids = torch.tensor([list(prompt)])
    input_ids = input_ids.to(model.device)

    if stop_token_ids is None:
        eos = model.generation_config.eos_token_id
        stop_token_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    stop_token_ids = set(stop_token_ids)

    cache = make_cache()
    _sync(model)
    start = time.perf_counter()
    logits = model(input_ids=input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
    _sync(model)
    prefill_seconds = time.perf_counter() - start

    new_tokens = []
    start = time.perf_counter()
    for _ in range(max_new_tokens):
        token = int(logits.argmax(dim=-1))
        if token in stop_token_ids:
            break
        new_tokens.append(token)
        if len(new_tokens) == max_new_tokens:
            break
        if stop_text is not None and stop_text in tokenizer.decode([token]):
            break
        next_input = torch.tensor([[token]], device=input_ids.device)
        logits = model(input_ids=next_input, past_key_values=cache, use_cache=True).logits[:, -1]
    _sync(model)
    decode_seconds = time.perf_counter() - start

    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if stop_text is not None:
        text = text.split(stop_text)[0]
    return {
        "text": text,
        "token_ids": new_tokens,
        "cache": cache,
        "prefill_seconds": prefill_seconds,
        "decode_seconds_per_token": decode_seconds / max(len(new_tokens), 1),
    }