import math
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import DynamicCache, PreTrainedModel


@torch.no_grad()
def perplexity(
    model: PreTrainedModel,
    chunks: torch.Tensor,
    batch_size: int = 1,
    cache_factory: Callable[[], DynamicCache] | None = None,
    stride: int | None = None,
    show_progress: bool = False,
) -> float:
    """Computes perplexity over fixed-length chunks (for example, from `data.load_wikitext`).

    Every token except the first in each chunk is predicted from the tokens before it, and
    perplexity is exp(total negative log-likelihood / number of predicted tokens).

    There are two ways to run it:
        - stride=None: one forward pass per chunk, like prefill. If `cache_factory` is
          given, its cache is used during that pass, so a fake-quantized cache affects the
          result. With no cache_factory, this is plain full-precision perplexity.
        - stride=k: feed each chunk k tokens at a time through the cache (teacher forcing).
          This matches how tokens arrive during decoding, so a cache's recent window
          behaves realistically, at the cost of more forward passes. stride=1 is
          exact token-by-token decoding.

    With no quantization, both modes should give the same perplexity up to floating-point
    noise, which is a good check that the evaluator itself is correct.

    Args:
        model: A causal LM.
        chunks: Token IDs of shape [num_chunks, chunk_len].
        batch_size: Number of chunks per forward pass.
        cache_factory: Function that returns a fresh cache for each batch, for example
            `lambda: FakeQuantKVCache(bits, window=128, config=model.config)`.
        stride: Tokens per forward pass in streaming mode. None means one pass per chunk.
        show_progress: Show a progress bar.

    Returns:
        The perplexity as a float.
    """
    starts = range(0, chunks.shape[0], batch_size)
    if show_progress:
        from tqdm.auto import tqdm

        starts = tqdm(starts, desc="perplexity")

    total_nll = 0.0
    total_tokens = 0
    for i in starts:
        batch = chunks[i : i + batch_size].to(model.device)
        if stride is None:
            cache = cache_factory() if cache_factory is not None else None
            logits = model(input_ids=batch, past_key_values=cache, use_cache=cache is not None).logits
            total_nll += _sum_nll(logits[:, :-1], batch[:, 1:])
        else:
            total_nll += _streamed_nll(model, batch, cache_factory, stride)
        total_tokens += batch.shape[0] * (batch.shape[1] - 1)

    return math.exp(total_nll / total_tokens)


def _streamed_nll(
    model: PreTrainedModel,
    batch: torch.Tensor,
    cache_factory: Callable[[], DynamicCache] | None,
    stride: int,
) -> float:
    """Total NLL of one batch, feeding `stride` tokens at a time through a cache."""
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}")
    cache = cache_factory() if cache_factory is not None else DynamicCache(config=model.config)

    seq_len = batch.shape[1]
    nll = 0.0
    # The last token is only ever a target, so inputs run over positions 0 .. seq_len - 2.
    for start in range(0, seq_len - 1, stride):
        end = min(start + stride, seq_len - 1)
        logits = model(input_ids=batch[:, start:end], past_key_values=cache, use_cache=True).logits
        nll += _sum_nll(logits, batch[:, start + 1 : end + 1])
    return nll


def _sum_nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Summed cross-entropy of logits [batch, n, vocab] against targets [batch, n], in float32."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="sum",
    ).item()