"""Real low-bit storage for KV tensors: values -> integer codes -> packed bytes, and back.

The rounding is the same asymmetric min-max scheme as `fake_quantize` in quant.py. The
difference is that the result is kept small:
    - codes: one integer from 0 to 2^bits - 1 per value, packed 8 // bits to a byte
    - scale and zero point: one FP16 pair per group of `group_size` values

Example with 2 bits, one group: [-1.0, 0.1, 0.5, 2.0]
    zero point (the minimum) = -1.0, scale (the step) = (2.0 - -1.0) / 3 = 1.0
    codes = round((x - zero) / scale) = [0, 1, 2, 3]    -> packed into one byte
    dequantized = codes * scale + zero = [-1.0, 0.0, 1.0, 2.0]
"""

from __future__ import annotations

import torch

SUPPORTED_BITS = (1, 2, 4, 8)


def quantize_to_codes(x: torch.Tensor, bits: int, group_size: int | None = None):
    """Rounds `x` onto a `bits`-bit grid, groups along the last dimension.

    Args:
        x: Tensor [..., D], e.g. keys [batch, heads, tokens, head_dim].
        bits: 1, 2, 4 or 8.
        group_size: Values per group. None uses the whole last dimension as one group.

    Returns:
        codes: uint8 [..., D], each between 0 and 2^bits - 1 (not packed yet).
        scale: float16 [..., D // group_size], the step size of each group.
        zero: float16 [..., D // group_size], the minimum of each group.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    last_dim = x.shape[-1]
    group_size = last_dim if group_size is None else group_size
    groups = x.reshape(*x.shape[:-1], last_dim // group_size, group_size).float()

    max_level = 2**bits - 1
    g_min = groups.amin(dim=-1, keepdim=True)
    g_max = groups.amax(dim=-1, keepdim=True)
    # Round the scale and zero point to FP16 first, because that's what gets stored, so the
    # codes are computed from exactly the numbers dequantize() will use.
    zero = g_min.half()
    # The floor keeps a constant group's scale above zero after rounding to FP16
    scale = ((g_max - g_min) / max_level).clamp(min=1e-6).half()

    codes = torch.clamp(torch.round((groups - zero.float()) / scale.float()), 0, max_level)
    return codes.to(torch.uint8).reshape(x.shape), scale.squeeze(-1), zero.squeeze(-1)


def dequantize(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Codes back to values: codes * scale + zero, per group.

    Args:
        codes: uint8 [..., D], unpacked.
        scale, zero: float16 [..., num_groups].
        dtype: Output dtype, e.g. the model's bfloat16.

    Returns:
        Tensor [..., D] in `dtype`.
    """
    num_groups = scale.shape[-1]
    groups = codes.reshape(*codes.shape[:-1], num_groups, -1).float()
    values = groups * scale.float().unsqueeze(-1) + zero.float().unsqueeze(-1)
    return values.reshape(codes.shape).to(dtype)


def _shifts(bits: int, device) -> torch.Tensor:
    return (torch.arange(8 // bits, device=device) * bits).to(torch.uint8)


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Packs codes [..., D] (uint8, each < 2^bits) into bytes [..., D * bits // 8].

    With 2 bits, four codes share a byte: the first in bits 0-1, the second in bits 2-3, and so on.
    """
    per_byte = 8 // bits
    if codes.shape[-1] % per_byte != 0:
        raise ValueError(f"The last dimension ({codes.shape[-1]}) must be a multiple of {per_byte}")
    grouped = codes.reshape(*codes.shape[:-1], -1, per_byte)
    # The shifted codes don't overlap, so adding them is the same as OR-ing them
    return (grouped << _shifts(bits, codes.device)).sum(dim=-1).to(torch.uint8)


def unpack(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Reverses `pack`: bytes [..., N] -> codes [..., N * 8 // bits] (uint8)."""
    mask = 2**bits - 1
    codes = (packed.unsqueeze(-1) >> _shifts(bits, packed.device)) & mask
    return codes.reshape(*packed.shape[:-1], -1)