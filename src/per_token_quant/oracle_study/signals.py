import torch
from transformers import PreTrainedModel

from quant import get_layer_kv


@torch.no_grad()
def compute_signals(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    head_reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    """Runs one prefill pass and computes all three signals.

    Args:
        model: A causal LM loaded with eager attention.
        input_ids: Token IDs of shape [batch, seq_len].
        attention_mask: Optional [batch, seq_len] mask (1 = real token, 0 = padding).
        head_reduction: "mean" averages over heads; "none" keeps a head dimension.

    Returns:
        A dict with "raw_attention", "normalized_attention", and "key_norm". Each is a
        float32 tensor of shape [num_layers, batch, seq_len] (or [num_layers, batch,
        heads, seq_len] with head_reduction="none"). Padded positions are set to 0.
    """
    input_ids = input_ids.to(model.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        output_attentions=True,
    )
    if outputs.attentions is None or any(a is None for a in outputs.attentions):
        raise ValueError("The model did not return attention weights. Load it with attn_implementation='eager'.")

    return {
        "raw_attention": accumulated_attention(outputs.attentions, attention_mask, False, head_reduction),
        "normalized_attention": accumulated_attention(outputs.attentions, attention_mask, True, head_reduction),
        "key_norm": key_l2_norm(outputs.past_key_values, attention_mask, head_reduction),
    }


def accumulated_attention(
    attentions: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor | None = None,
    normalize: bool = False,
    head_reduction: str = "mean",
) -> torch.Tensor:
    """Total attention each token receives, optionally normalized by the number of queries.

    Assumes queries and keys are the same positions, as in a single prefill pass with no
    earlier cache.

    Args:
        attentions: One tensor per layer, each [batch, heads, seq_len, seq_len], laid out
            as [query, key] in the last two dimensions.
        attention_mask: Optional [batch, seq_len] mask (1 = real token, 0 = padding).
        normalize: If True, divide each token's total by the number of real queries that
            can attend to it (itself and every later token), as in ZipCache.
        head_reduction: "mean" averages over heads; "none" keeps them.

    Returns:
        Float32 tensor [num_layers, batch, seq_len], or [num_layers, batch, heads, seq_len].
    """
    _check_head_reduction(head_reduction)
    batch, _, seq_len, _ = attentions[0].shape
    device = attentions[0].device
    mask = (
        torch.ones(batch, seq_len, device=device)
        if attention_mask is None
        else attention_mask.to(device=device, dtype=torch.float32)
    )

    if normalize:
        # For key position k, count the real query positions q >= k (reverse cumulative sum).
        num_queries = mask.flip(-1).cumsum(-1).flip(-1).clamp(min=1)  # [batch, seq_len]

    scores = []
    for attn in attentions:
        weights = attn.float() * mask[:, None, :, None]  # drop rows from padded queries
        received = weights.sum(dim=2)  # [batch, heads, seq_len]: summed over queries
        if normalize:
            received = received / num_queries[:, None, :]
        received = received * mask[:, None, :]  # padded keys get 0
        scores.append(received.mean(dim=1) if head_reduction == "mean" else received)
    return torch.stack(scores)


def key_l2_norm(cache, attention_mask: torch.Tensor | None = None, head_reduction: str = "mean") -> torch.Tensor:
    """L2 norm of every cached key vector.

    The cache stores keys after RoPE, but RoPE is a rotation, so it doesn't change the norm.

    Args:
        cache: The cache returned by the forward pass (`outputs.past_key_values`).
        attention_mask: Optional [batch, seq_len] mask (1 = real token, 0 = padding).
        head_reduction: "mean" averages over KV heads; "none" keeps them.

    Returns:
        Float32 tensor [num_layers, batch, seq_len], or [num_layers, batch, kv_heads, seq_len].
        Lower values suggest more important tokens.
    """
    _check_head_reduction(head_reduction)
    norms = []
    for layer_idx in range(len(cache)):
        keys, _ = get_layer_kv(cache, layer_idx)
        norm = keys.float().norm(dim=-1)  # [batch, kv_heads, seq_len]
        if attention_mask is not None:
            norm = norm * attention_mask.to(device=norm.device, dtype=torch.float32)[:, None, :]
        norms.append(norm.mean(dim=1) if head_reduction == "mean" else norm)
    return torch.stack(norms)


def _check_head_reduction(head_reduction: str) -> None:
    if head_reduction not in ("mean", "none"):
        raise ValueError(f"head_reduction must be 'mean' or 'none', got {head_reduction!r}")