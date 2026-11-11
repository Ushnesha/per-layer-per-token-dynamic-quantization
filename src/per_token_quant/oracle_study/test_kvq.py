import math

import torch
from transformers import AutoModelForCausalLM, DynamicCache, LlamaConfig

from metrics import perplexity
from quant import FULL_PRECISION, FakeQuantKVCache, fake_quantize, fake_quantize_tokens, get_layer_kv
from signals import compute_signals

SEQ_LEN = 24


def tiny_model():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA, like Llama 3
        max_position_embeddings=128,
    )
    return AutoModelForCausalLM.from_config(config, attn_implementation="eager", dtype=torch.float32).eval()


def random_ids(batch=2, seq_len=SEQ_LEN):
    return torch.randint(0, 100, (batch, seq_len), generator=torch.Generator().manual_seed(1))


def logits_with(model, ids, cache):
    with torch.no_grad():
        return model(input_ids=ids, past_key_values=cache, use_cache=True).logits


# ---------------------------------------------------------------- fake_quantize


def test_fake_quantize_full_precision_is_identity():
    x = torch.randn(2, 3, 5, 16)
    assert fake_quantize(x, FULL_PRECISION) is x


def test_fake_quantize_levels_range_and_error():
    x = torch.randn(2, 3, 5, 16)
    errors = []
    for bits in (8, 4, 2):
        q = fake_quantize(x, bits, group_size=8)
        groups, qg = x.reshape(-1, 8), q.reshape(-1, 8)
        # at most 2^bits distinct values per group, all inside the group's [min, max]
        assert all(len(torch.unique(row)) <= 2**bits for row in qg)
        assert (qg >= groups.amin(-1, keepdim=True) - 1e-5).all()
        assert (qg <= groups.amax(-1, keepdim=True) + 1e-5).all()
        errors.append((q - x).abs().mean().item())
    assert errors[0] < errors[1] < errors[2]  # fewer bits, more error


def test_fake_quantize_constant_group():
    x = torch.full((1, 1, 2, 8), 3.0)
    assert torch.equal(fake_quantize(x, 4), x)


# ---------------------------------------------------------------- FakeQuantKVCache


def test_cache_matches_dynamic_cache_when_quantization_is_off():
    model, ids = tiny_model(), random_ids()
    reference = logits_with(model, ids, DynamicCache(config=model.config))
    all_full = torch.full((SEQ_LEN,), FULL_PRECISION)
    for window in (0, 4):
        out = logits_with(model, ids, FakeQuantKVCache(all_full, window=window, config=model.config))
        assert torch.equal(out, reference)


def test_quantizing_one_token_only_affects_later_positions():
    model, ids = tiny_model(), random_ids()
    j = 10
    bits = torch.full((SEQ_LEN,), FULL_PRECISION)
    bits[j] = 2
    reference = logits_with(model, ids, DynamicCache(config=model.config))
    out = logits_with(model, ids, FakeQuantKVCache(bits, config=model.config))
    diff = (out - reference).abs()
    assert diff[:, :j].max() < 1e-6  # causal: earlier positions can't see token j
    assert diff[:, j:].max() > 1e-6  # token j and later do


def test_window_keeps_recent_tokens_full_precision():
    """Old tokens get quantized and the last `window` tokens don't, in prefill and decoding."""
    model, ids = tiny_model(), random_ids(batch=1)
    window = 4
    bits = torch.full((SEQ_LEN,), 4)
    bits[:2] = FULL_PRECISION  # two sink tokens
    final_bits = bits.clone()
    final_bits[-window:] = FULL_PRECISION  # the last `window` tokens are never quantized

    def run(cache, streaming):
        if streaming:
            for t in range(SEQ_LEN):
                logits_with(model, ids[:, t : t + 1], cache)
        else:
            logits_with(model, ids, cache)
        return cache

    for streaming in (False, True):
        # Layer-0 keys only depend on the input tokens, so the full-precision run gives
        # exactly the keys the quantized cache started from.
        full_keys, _ = get_layer_kv(run(DynamicCache(config=model.config), streaming), 0)
        cache = run(FakeQuantKVCache(bits, window=window, config=model.config), streaming)
        keys, _ = get_layer_kv(cache, 0)
        assert torch.equal(keys, fake_quantize_tokens(full_keys, final_bits[None])), streaming
        assert cache.num_done[0] == SEQ_LEN - window


# ---------------------------------------------------------------- perplexity


def test_streaming_perplexity_matches_prefill():
    model, chunks = tiny_model(), random_ids(batch=3)
    full = perplexity(model, chunks, batch_size=2)
    for stride in (1, 5):
        streamed = perplexity(model, chunks, batch_size=2, stride=stride)
        assert math.isclose(full, streamed, rel_tol=1e-4), (stride, full, streamed)


def test_perplexity_with_quantized_cache_runs():
    model, chunks = tiny_model(), random_ids(batch=2)
    bits = torch.full((SEQ_LEN,), 2)
    bits[:2] = FULL_PRECISION
    factory = lambda: FakeQuantKVCache(bits, window=4, config=model.config)
    for stride in (None, 3):
        ppl = perplexity(model, chunks, cache_factory=factory, stride=stride)
        assert math.isfinite(ppl) and ppl > 1


# ---------------------------------------------------------------- signals


def test_signals_shapes_and_sums():
    model, ids = tiny_model(), random_ids()
    signals = compute_signals(model, ids)
    for name in ("raw_attention", "normalized_attention", "key_norm"):
        assert signals[name].shape == (2, 2, SEQ_LEN), name

    raw, normalized = signals["raw_attention"], signals["normalized_attention"]
    # every query's attention sums to 1, so the totals over keys equal the number of queries
    assert torch.allclose(raw.sum(-1), torch.full((2, 2), float(SEQ_LEN)), atol=1e-3)
    # token k can be attended by SEQ_LEN - k queries
    num_queries = torch.arange(SEQ_LEN, 0, -1, dtype=torch.float32)
    assert torch.allclose(normalized, raw / num_queries, atol=1e-5)
    assert (signals["key_norm"] > 0).all()


def test_signals_with_padding():
    model, ids = tiny_model(), random_ids()
    mask = torch.ones_like(ids)
    mask[0, :5] = 0  # left padding on the first sequence
    signals = compute_signals(model, ids, attention_mask=mask)
    for name, value in signals.items():
        assert (value[:, 0, :5] == 0).all(), name


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"all {len(tests)} tests passed")