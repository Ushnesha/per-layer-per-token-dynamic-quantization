"""Sanity tests for adaptive_kv.py (prototype v0), on the tiny random model from test_kvq.py.

Run from the folder that holds adaptive_kv.py with:

    python -m pytest test_adaptive_kv.py -q
or, without pytest:
    python test_adaptive_kv.py
"""

import os
import tempfile

import torch
import torch.nn as nn
from transformers import DynamicCache

from .adaptive_kv import AdaptiveKVCache, KeyNormLinearScorer, PositionScorer, RandomScorer, generate
from l2_norm_mlp.policies import key_norm_linear_scores
from oracle.quant import FULL_PRECISION, fake_quantize_tokens, get_layer_kv
from oracle.test_kvq import SEQ_LEN, random_ids, tiny_model
from l2_norm_mlp.test_policies import FakeTokenizer

WINDOW = 4
SINKS = 2


def plain_greedy(model, ids, num_new):
    """Greedy decoding with a normal cache, for comparison."""
    tokens = []
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for _ in range(num_new):
            tokens.append(int(logits.argmax(dim=-1)))
            logits = model(input_ids=torch.tensor([tokens[-1:]]), past_key_values=cache, use_cache=True).logits[:, -1]
    return tokens


def run(model, ids, scorer, budget, num_new=3, low_bits=2):
    make_cache = lambda: AdaptiveKVCache(
        scorer, budget=budget, low_bits=low_bits, window=WINDOW, num_sinks=SINKS, config=model.config
    )
    return generate(model, FakeTokenizer(), ids[0].tolist(), make_cache, max_new_tokens=num_new, stop_token_ids=[])


def save_layer_picking_checkpoint(model, folder, layer):
    """A key-norms-only checkpoint whose score is just one layer's key norm."""
    num_layers = model.config.num_hidden_layers
    linear = nn.Linear(num_layers, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, layer] = 1.0
        linear.bias.zero_()
    path = os.path.join(folder, "ckpt.pt")
    torch.save(
        {
            "state_dict": linear.state_dict(),
            "mean": torch.zeros(num_layers),
            "std": torch.ones(num_layers),
            "use_position": False,
            "llm": model.name_or_path,
        },
        path,
    )
    return path


def test_keeping_everything_matches_plain_greedy():
    model, ids = tiny_model(), random_ids(batch=1)
    out = run(model, ids, PositionScorer(), budget=1.0, num_new=5)
    assert out["token_ids"] == plain_greedy(model, ids, 5)
    assert (out["cache"].bits == FULL_PRECISION).all()


def test_prefill_is_full_precision_and_old_tokens_get_quantized():
    model, ids = tiny_model(), random_ids(batch=1)
    out = run(model, ids, PositionScorer(), budget=0.0, num_new=3)
    cache = out["cache"]

    # The first token comes from the full-precision prefill
    assert out["token_ids"][0] == plain_greedy(model, ids, 1)[0]

    # 3 new tokens: 2 are fed back, so tokens up to SEQ_LEN + 2 - WINDOW have been decided
    decided = SEQ_LEN + 2 - WINDOW
    assert cache.num_decided == decided
    expected_bits = torch.full((decided,), 2, dtype=torch.long)
    expected_bits[:SINKS] = FULL_PRECISION
    assert torch.equal(cache.bits, expected_bits)

    # Layer-0 keys only depend on the input tokens, so they can be checked exactly
    with torch.no_grad():
        plain = DynamicCache(config=model.config)
        model(input_ids=ids, past_key_values=plain, use_cache=True)
    full_keys, _ = get_layer_kv(plain, 0)
    keys, _ = get_layer_kv(cache, 0)
    expected = fake_quantize_tokens(full_keys[:, :, :decided], expected_bits[None])
    assert torch.allclose(keys[:, :, :decided], expected, atol=1e-6)


def test_budget_keeps_the_top_fraction():
    model, ids = tiny_model(), random_ids(batch=1)
    out = run(model, ids, PositionScorer(), budget=0.5, num_new=2)
    bits = out["cache"].bits
    # 1 token fed back: the first decision covers positions 0 .. SEQ_LEN - WINDOW, sinks excluded
    competing = SEQ_LEN + 1 - WINDOW - SINKS
    k = int(round(0.5 * competing))
    # Position scores favor earlier tokens, so the earliest k competing tokens stay
    assert (bits[SINKS : SINKS + k] == FULL_PRECISION).all()
    assert (bits[SINKS + k :] == 2).all()


def test_inline_key_norm_scores_match_the_offline_policy():
    """The cache's own key norms give the same scores as policies.key_norm_linear_scores."""
    model, ids = tiny_model(), random_ids(batch=1)
    with tempfile.TemporaryDirectory() as folder:
        path = save_layer_picking_checkpoint(model, folder, layer=1)
        out = run(model, ids, KeyNormLinearScorer(path), budget=0.5, num_new=2)
        offline = key_norm_linear_scores(path)(model, ids)[0]
    cache = out["cache"]
    assert torch.allclose(cache.scores, offline[: cache.num_decided], atol=1e-5)


def test_memory_report():
    model, ids = tiny_model(), random_ids(batch=1)
    full = run(model, ids, PositionScorer(), budget=1.0)["cache"].memory_report()
    assert full["compression"] == 1.0 and full["low_bit_tokens"] == 0

    low = run(model, ids, PositionScorer(), budget=0.0)["cache"].memory_report()
    assert low["low_bit_tokens"] == SEQ_LEN + 2 - WINDOW - SINKS
    assert low["compression"] > 1.0 and low["average_bits_per_value"] < 16


def test_random_scorer_is_reproducible():
    norms, positions = torch.randn(10, 2), torch.arange(10)
    assert torch.equal(RandomScorer(5)(norms, positions), RandomScorer(5)(norms, positions))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"all {len(tests)} tests passed")