"""Sanity tests for real_kv.py (prototype v1), on the tiny random model from test_kvq.py.

Run from src/per_token_quant with:
    python -m prototype.test_real_kv
"""

import torch

from l2_norm_mlp.test_policies import FakeTokenizer
from oracle.quant import FULL_PRECISION, get_layer_kv
from oracle.test_kvq import random_ids, tiny_model
from prototype.adaptive_kv import AdaptiveKVCache, PositionScorer, generate
from prototype.real_kv import RealQuantKVCache
from prototype.test_adaptive_kv import SINKS, WINDOW, plain_greedy

EXTRA_TOKENS = [1, 2, 3]  # fed one at a time after the prompt, the same for v0 and v1


def make_cache(cache_class, model, budget, low_bits=2):
    return cache_class(
        PositionScorer(), budget=budget, low_bits=low_bits, window=WINDOW, num_sinks=SINKS, config=model.config
    )


def feed(model, ids, cache):
    """The prompt in one pass, then EXTRA_TOKENS one at a time. Returns the last logits."""
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
        for token in EXTRA_TOKENS:
            logits = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits
    return logits


def test_keeping_everything_matches_plain_greedy():
    model, ids = tiny_model(), random_ids(batch=1)
    make = lambda: make_cache(RealQuantKVCache, model, budget=1.0)
    out = generate(model, FakeTokenizer(), ids[0].tolist(), make, max_new_tokens=5, stop_token_ids=[])
    assert out["token_ids"] == plain_greedy(model, ids, 5)
    assert out["cache"].low_positions.numel() == 0


def test_v1_matches_v0():
    model, ids = tiny_model(), random_ids(batch=1)
    v0, v1 = make_cache(AdaptiveKVCache, model, 0.5), make_cache(RealQuantKVCache, model, 0.5)
    logits0, logits1 = feed(model, ids, v0), feed(model, ids, v1)

    # Same decisions
    assert torch.equal(v0.bits, v1.bits)
    assert torch.equal(v1.low_positions, (v1.bits < FULL_PRECISION).nonzero().squeeze(-1))
    assert v1.get_seq_length() == v0.get_seq_length() == ids.shape[1] + len(EXTRA_TOKENS)

    # Nearly the same values (FP16 scales in v1, FP32 in v0), and so nearly the same outputs
    keys0, values0 = get_layer_kv(v0, 0)
    keys1, values1 = v1.layer_kv(0)
    assert keys1.shape == keys0.shape
    assert (keys1 - keys0).abs().mean() < 1e-2 and (values1 - values0).abs().mean() < 1e-2
    assert (logits1 - logits0).abs().mean() < 1e-2


def test_low_bit_tokens_are_really_stored_small():
    model, ids = tiny_model(), random_ids(batch=1)
    cache = make_cache(RealQuantKVCache, model, budget=0.0)
    feed(model, ids, cache)

    num_low = cache.low_positions.numel()
    num_full = cache.num_tokens - num_low
    assert num_low == cache.num_decided - SINKS  # every decided token except the sinks
    for layer in range(cache.num_model_layers):
        assert cache.full_keys[layer].shape[-2] == num_full  # really removed from the full-precision pile
        assert cache.low_keys[layer]["codes"].dtype == torch.uint8
        assert cache.low_keys[layer]["codes"].shape[-2] == num_low

    values_per_token = 2 * cache.num_model_layers * cache.kv_heads * cache.head_dim
    full_value_bytes = cache.full_keys[0].element_size()  # 4 for the FP32 test model, 2 for BF16
    groups_per_token = values_per_token // cache.head_dim  # group_size None: one group per head
    expected = num_full * values_per_token * full_value_bytes + num_low * (
        values_per_token * 2 // 8 + groups_per_token * 2 * 2  # 2-bit codes + FP16 scale and zero point
    )
    assert cache.stored_bytes() == expected
    assert cache.stored_bytes() < cache.num_tokens * values_per_token * full_value_bytes


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"all {len(tests)} tests passed")