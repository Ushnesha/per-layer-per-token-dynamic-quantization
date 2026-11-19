"""Sanity tests for packing.py. Pure tensors, runs on CPU in a second.

Run from src/per_token_quant with:
    python -m prototype.test_packing
"""

import torch

from oracle.quant import fake_quantize
from prototype.packing import dequantize, pack, quantize_to_codes, unpack


def test_the_docstring_example():
    x = torch.tensor([[-1.0, 0.1, 0.5, 2.0]])
    codes, scale, zero = quantize_to_codes(x, bits=2)
    assert codes.tolist() == [[0, 1, 2, 3]]
    assert pack(codes, 2).tolist() == [[0 | 1 << 2 | 2 << 4 | 3 << 6]]  # one byte: 228
    assert torch.allclose(dequantize(codes, scale, zero, torch.float32), torch.tensor([[-1.0, 0.0, 1.0, 2.0]]))


def test_pack_then_unpack_gives_back_the_codes():
    for bits in (1, 2, 4, 8):
        codes = torch.randint(0, 2**bits, (2, 3, 5, 16), dtype=torch.uint8)
        packed = pack(codes, bits)
        assert packed.dtype == torch.uint8 and packed.shape[-1] == 16 * bits // 8
        assert torch.equal(unpack(packed, bits), codes)


def test_dequantized_values_match_fake_quantize():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 64)
    for bits, group_size in ((2, 32), (2, None), (4, 16)):
        codes, scale, zero = quantize_to_codes(x, bits, group_size)
        assert int(codes.max()) <= 2**bits - 1
        rebuilt = dequantize(codes, scale, zero, torch.float32)

        # Every value stays within half a step of the original
        step = scale.float().repeat_interleave(x.shape[-1] // scale.shape[-1], dim=-1)
        assert ((rebuilt - x).abs() <= step / 2 + 1e-3).all()

        # Nearly the same as v0's fake quantization. The FP16 scale and zero point make tiny
        # differences, and once in a while a value right on a rounding boundary lands one level over.
        assert (rebuilt - fake_quantize(x, bits, group_size)).abs().mean() < 1e-2


def test_constant_group_survives():
    x = torch.full((1, 8), 3.0)
    codes, scale, zero = quantize_to_codes(x, bits=2)
    assert torch.allclose(dequantize(codes, scale, zero, torch.float32), x)


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"all {len(tests)} tests passed")