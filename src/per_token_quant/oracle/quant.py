import torch
from transformers import DynamicCache

FULL_PRECISION = 16


def fake_quantize(x: torch.Tensor, bits: int, group_size: int | None = None) -> torch.Tensor:
    """Asymmetric min-max fake quantization along the last dimension.

    The last dimension is split into groups of `group_size` elements, and each group gets
    its own scale and minimum. For a KV tensor [batch, heads, seq_len, head_dim], that means
    every token in every head is quantized on its own.

    Args:
        x: Tensor to quantize.
        bits: Bit-width. 16 or more returns `x` unchanged.
        group_size: Elements per group. None uses the whole last dimension as one group.

    Returns:
        A tensor with the same shape and dtype as `x`.
    """
    if bits >= FULL_PRECISION:
        return x
    
    last_dim = x.shape[-1]
    group_size = last_dim if group_size is None else group_size
    groups = x.reshape(*x.shape[:-1], last_dim // group_size, group_size).float()

    g_min = groups.amin(dim=-1, keepdim=True)
    g_max = groups.amax(dim=-1, keepdim=True)
    max_level = 2**bits - 1
    scale = (g_max - g_min).clamp(min=1e-8) / max_level

    levels = torch.clamp(torch.round((groups - g_min) / scale), 0, max_level)
    return (levels * scale + g_min).reshape(x.shape).to(x.dtype)
    

def fake_quantize_tokens(x: torch.Tensor, token_bits: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Fake-quantizes each token of a KV tensor at its own bit-width.

    Args:
        x: Key or value tensor [batch, heads, seq_len, head_dim].
        token_bits: Bit-width per token, [batch, seq_len] (or [1, seq_len] for the same map
            on every sequence). 16 means leave the token alone.
        group_size: Passed to `fake_quantize`.
    """
    out = x
    for bit in torch.unique(token_bits).tolist():
        if bit < FULL_PRECISION:
            mask = (token_bits == bit)[:, None, :, None]  # broadcast over heads and head_dim
            out = torch.where(mask, fake_quantize(x, bit, group_size), out)
    return out


def get_layer_kv(cache: DynamicCache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns one layer's stored (keys, values), each [batch, kv_heads, seq_len, head_dim]."""
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


class FakeQuantKVCache(DynamicCache):
    """A DynamicCache that fake-quantizes each token according to a fixed bit map.

    `bits[j]` (or `bits[b, j]` for a per-sequence map) is the bit-width of token j, with 16
    meaning full precision. Token j is quantized once, as soon as it is more than `window`
    tokens old, and stays quantized. This happens inside `update`, so attention in the same
    forward pass already uses the quantized keys and values.

    Examples:
        Oracle, quantize only token j to INT4:
            bits = torch.full((seq_len,), 16); bits[j] = 4
            cache = FakeQuantKVCache(bits, config=model.config)

        Uniform INT4, keeping 4 sink tokens and the last 128 tokens at FP16:
            bits = torch.full((seq_len,), 4); bits[:4] = 16
            cache = FakeQuantKVCache(bits, window=128, config=model.config)

    In a single prefill pass, every token older than the window is quantized before the
    prefill's attention runs. To see the window behave as it would during decoding, feed
    tokens a few at a time (`metrics.perplexity` with `stride`).

    Args:
        bits: Integer tensor [seq_len] or [batch, seq_len]. Tokens past its end are never
            quantized.
        window: Number of most recent tokens kept at full precision.
        group_size: Group size along head_dim. None uses the whole head_dim.
        quant_layers: If given, only these layer indices are quantized.
        **kwargs: Passed to DynamicCache, e.g. `config=model.config`.
    """

    def __init__(
        self,
        bits: torch.Tensor,
        window: int = 0,
        group_size: int | None = None,
        quant_layers: set[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bits = bits.long().reshape(-1, bits.shape[-1])  # always [batch or 1, seq_len]
        self.window = window
        self.group_size = group_size
        self.quant_layers = quant_layers  # not `layers`: DynamicCache uses that name
        self.num_done: dict[int, int] = {}  # layer_idx -> tokens already handled

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        keys, values = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        if self.quant_layers is not None and layer_idx not in self.quant_layers:
            return keys, values

        start = self.num_done.get(layer_idx, 0)
        end = min(keys.shape[-2] - self.window, self.bits.shape[1])
        if end > start:
            token_bits = self.bits[:, start:end].to(keys.device)
            if (token_bits < FULL_PRECISION).any():
                # `keys` and `values` are the cache's own storage, so this edits the cache.
                keys[:, :, start:end] = fake_quantize_tokens(keys[:, :, start:end], token_bits, self.group_size)
                values[:, :, start:end] = fake_quantize_tokens(values[:, :, start:end], token_bits, self.group_size)
            self.num_done[layer_idx] = end
        return keys, values