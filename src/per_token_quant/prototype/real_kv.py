"""Prototype v1: the adaptive KV cache with real low-bit storage.

The decisions are exactly v0's (`AdaptiveKVCache`): same scorer, budget, threshold, window
and sinks. What changes is how tokens are stored. Every layer keeps two piles:

    - full precision: the sinks, the tokens the policy kept, and the recent window, in the
      model's dtype (BF16 for Llama)
    - low bit: every other token, as packed codes plus an FP16 scale and zero point per
      group (see packing.py)

When tokens leave the window, the low-bit ones are packed into the low-bit pile and removed
from the full-precision pile, so the memory really shrinks. Attention still needs a normal
tensor, so right before a layer's attention runs, its keys and values are rebuilt: the
low-bit pile is unpacked and every token goes back to its position. That rebuilt copy only
lives while that one layer runs.

v1 saves memory, not time. Rebuilding every layer at every step makes decoding slower than
v0. Reading the packed codes directly inside the attention kernel (v2) is what makes it fast.

`stored_bytes()` adds up the bytes the two piles really hold. With a BF16 model it equals
`memory_report()["kv_bytes"]`, the memory v0 predicted.

Batch size 1 only. Usage is the same as v0, swapping the class:
    make_cache = lambda: RealQuantKVCache(scorer, budget=0.1, low_bits=2, group_size=32, config=model.config)
    out = generate(model, tokenizer, prompt, make_cache)
"""

from __future__ import annotations

import torch

from oracle.quant import FULL_PRECISION
from prototype.adaptive_kv import AdaptiveKVCache
from prototype.packing import SUPPORTED_BITS, dequantize, pack, quantize_to_codes, unpack


class RealQuantKVCache(AdaptiveKVCache):
    """`AdaptiveKVCache` with the low-bit tokens really stored in `low_bits` bits.

    Takes the same arguments as `AdaptiveKVCache`. `low_bits` must be 1, 2, 4 or 8.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.low_bits not in SUPPORTED_BITS:
            raise ValueError(f"low_bits must be one of {SUPPORTED_BITS}, got {self.low_bits}")
        layers = self.num_model_layers
        # Full-precision pile per layer: [1, kv_heads, tokens, head_dim], ordered by position
        self.full_keys: list[torch.Tensor | None] = [None] * layers
        self.full_values: list[torch.Tensor | None] = [None] * layers
        # Low-bit pile per layer: {"codes": packed uint8, "scale": float16, "zero": float16}
        self.low_keys: list[dict | None] = [None] * layers
        self.low_values: list[dict | None] = [None] * layers
        self.layer_lengths = [0] * layers  # tokens stored in each layer
        # Positions of the decided tokens, the same in every layer, in increasing order
        self.kept_positions = torch.empty(0, dtype=torch.long)  # kept in full precision (sinks included)
        self.low_positions = torch.empty(0, dtype=torch.long)  # stored in low bits

    # ------------------------------------------------------------ storage (replaces v0's)

    def _store(self, key_states, value_states, layer_idx, *args, **kwargs):
        """Adds the new tokens to the full-precision pile, then rebuilds the layer for attention."""
        if self.full_keys[layer_idx] is None:
            self.full_keys[layer_idx], self.full_values[layer_idx] = key_states, value_states
        else:
            self.full_keys[layer_idx] = torch.cat((self.full_keys[layer_idx], key_states), dim=-2)
            self.full_values[layer_idx] = torch.cat((self.full_values[layer_idx], value_states), dim=-2)
        self.layer_lengths[layer_idx] += key_states.shape[-2]
        return self.layer_kv(layer_idx)

    def _quantize(self, start: int, end: int, bits: torch.Tensor) -> None:
        """Moves the low-bit tokens among start .. end - 1 from the full-precision pile to the low-bit pile.

        Runs at the start of a forward pass, before any layer stores this pass's tokens. Tokens
        start .. end - 1 are the oldest undecided ones, so in every layer's full-precision pile
        they sit right after the tokens already kept.
        """
        keep_idx = (bits >= FULL_PRECISION).nonzero().squeeze(-1)  # offsets within start .. end - 1
        low_idx = (bits < FULL_PRECISION).nonzero().squeeze(-1)
        num_kept_before = self.kept_positions.numel()
        num_new = end - start

        if low_idx.numel() > 0:
            for layer in range(self.num_model_layers):
                for full_pile, low_pile in ((self.full_keys, self.low_keys), (self.full_values, self.low_values)):
                    full = full_pile[layer]
                    block = full[:, :, num_kept_before : num_kept_before + num_new]

                    # Pack the low-bit tokens and add them to the low-bit pile
                    codes, scale, zero = quantize_to_codes(
                        block[:, :, low_idx.to(full.device)], self.low_bits, self.group_size
                    )
                    new_low = {"codes": pack(codes, self.low_bits), "scale": scale, "zero": zero}
                    if low_pile[layer] is None:
                        low_pile[layer] = new_low
                    else:
                        low_pile[layer] = {name: torch.cat((low_pile[layer][name], new_low[name]), dim=-2) for name in new_low}

                    # Remove them from the full-precision pile: this is where the memory is freed
                    stay = torch.cat(
                        (
                            torch.arange(num_kept_before),
                            num_kept_before + keep_idx,
                            torch.arange(num_kept_before + num_new, full.shape[-2]),
                        )
                    ).to(full.device)
                    full_pile[layer] = full.index_select(-2, stay)

        self.kept_positions = torch.cat((self.kept_positions, start + keep_idx))
        self.low_positions = torch.cat((self.low_positions, start + low_idx))

    def layer_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuilds one layer's full keys and values [1, kv_heads, tokens, head_dim] for attention."""
        full_keys, full_values = self.full_keys[layer_idx], self.full_values[layer_idx]
        if self.low_positions.numel() == 0:
            return full_keys, full_values  # nothing in low bits: the full-precision pile is the whole layer

        device = full_keys.device
        length = self.layer_lengths[layer_idx]
        # The full-precision pile holds the kept tokens, then every undecided token in order
        full_positions = torch.cat((self.kept_positions, torch.arange(self.num_decided, length))).to(device)
        low_positions = self.low_positions.to(device)

        rebuilt = []
        for full, low in ((full_keys, self.low_keys[layer_idx]), (full_values, self.low_values[layer_idx])):
            out = full.new_empty((full.shape[0], full.shape[1], length, full.shape[-1]))
            out[:, :, full_positions] = full
            codes = unpack(low["codes"], self.low_bits)
            out[:, :, low_positions] = dequantize(codes, low["scale"], low["zero"], full.dtype)
            rebuilt.append(out)
        return rebuilt[0], rebuilt[1]

    # ------------------------------------------------------------ lengths for the model

    # The tokens live in the two piles, not in DynamicCache's own storage, so the cache has to
    # report its length itself. The model uses these to number the new tokens and size the mask.

    def get_seq_length(self, layer_idx: int = 0, *args, **kwargs) -> int:
        return self.layer_lengths[layer_idx] if layer_idx < len(self.layer_lengths) else 0

    def get_mask_sizes(self, cache_position, layer_idx: int = 0, *args, **kwargs) -> tuple[int, int]:
        query_length = cache_position.shape[0] if torch.is_tensor(cache_position) else int(cache_position)
        return self.get_seq_length(layer_idx) + query_length, 0

    # ------------------------------------------------------------ memory

    def stored_bytes(self) -> int:
        """Bytes the cache really holds: both piles, keys and values, every layer."""
        total = 0
        for layer in range(self.num_model_layers):
            for tensor in (self.full_keys[layer], self.full_values[layer]):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
            for low in (self.low_keys[layer], self.low_values[layer]):
                if low is not None:
                    total += sum(t.numel() * t.element_size() for t in low.values())
        return total