"""Sanity tests for policies.py, on the same tiny random model as test_kvq.py.

Run from the folder that holds policies.py with:

    python -m pytest test_policies.py -q
or, without pytest:
    python test_policies.py
"""

import math
import os
import tempfile

import torch
import torch.nn as nn

from transformers import DynamicCache

from oracle.metrics import perplexity
from .policies import (
    build_bit_map,
    key_norm_linear_scores,
    policy_passkey,
    policy_perplexity,
    position_scores,
    random_scores,
    windowed_attention_scores,
)
from oracle.quant import FULL_PRECISION
from oracle.signals import compute_signals, key_l2_norm
from oracle.test_kvq import SEQ_LEN, random_ids, tiny_model


class FakeTokenizer:
    """The tiny model has no tokenizer, so token IDs are 'decoded' as space-separated numbers."""

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_bit_map_keeps_sinks_and_top_scores():
    scores = torch.tensor([[0.0, 5.0, 1.0, 4.0, 3.0, 2.0]])
    bits = build_bit_map(scores, budget=0.4, low_bits=2, num_sinks=1)
    # 5 non-sink tokens, 40% -> 2 kept: the scores 5.0 and 4.0 at positions 1 and 3
    assert bits.tolist() == [[16, 16, 2, 16, 2, 2]]


def test_bit_map_ignores_tokens_that_never_leave_the_window():
    scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 9.0, 9.0]])
    bits = build_bit_map(scores, budget=1 / 3, low_bits=2, num_sinks=1, window=2)
    # positions 4 and 5 stay in the window, so only 1-3 compete: 1 of 3 kept, the 3.0
    assert bits.tolist() == [[16, 2, 2, 16, 16, 16]]


def test_bit_map_budget_extremes():
    scores = torch.randn(2, 10)
    assert (build_bit_map(scores, 1.0, low_bits=2) == FULL_PRECISION).all()
    none_kept = build_bit_map(scores, 0.0, low_bits=2, num_sinks=4)
    assert (none_kept[:, :4] == FULL_PRECISION).all() and (none_kept[:, 4:] == 2).all()


def test_all_fp16_matches_plain_perplexity():
    model, chunks = tiny_model(), random_ids(batch=2)
    plain = perplexity(model, chunks, stride=3)
    ours = policy_perplexity(model, chunks, None, budget=1.0, low_bits=2, window=4, num_sinks=2, stride=3)
    assert math.isclose(plain, ours, rel_tol=1e-6), (plain, ours)


def test_uniform_low_bit_changes_perplexity():
    model, chunks = tiny_model(), random_ids(batch=2)
    full = policy_perplexity(model, chunks, None, budget=1.0, low_bits=2, window=4, num_sinks=2, stride=3)
    low = policy_perplexity(model, chunks, None, budget=0.0, low_bits=2, window=4, num_sinks=2, stride=3)
    assert math.isfinite(low) and low != full


def test_windowed_attention_without_window_matches_raw_attention():
    model, ids = tiny_model(), random_ids()
    raw = compute_signals(model, ids)["raw_attention"].mean(dim=0)  # [batch, seq_len]
    ours = windowed_attention_scores(window=None)(model, ids)
    assert torch.allclose(ours, raw, atol=1e-5)


def test_windowed_attention_window_one_is_the_diagonal():
    model, ids = tiny_model(), random_ids()
    with torch.no_grad():
        attentions = model(input_ids=ids, output_attentions=True).attentions
    diagonal = torch.stack([a.diagonal(dim1=-2, dim2=-1).mean(dim=1) for a in attentions]).mean(dim=0)
    ours = windowed_attention_scores(window=1)(model, ids)
    assert torch.allclose(ours, diagonal, atol=1e-6)


def test_linear_scores_rebuild_the_training_features():
    """A checkpoint whose weights pick out layer 0 must return layer 0's key norms."""
    model, ids = tiny_model(), random_ids()
    num_layers = model.config.num_hidden_layers
    linear = nn.Linear(num_layers + 1, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = 1.0
        linear.bias.zero_()
    ckpt = {
        "state_dict": linear.state_dict(),
        "mean": torch.zeros(num_layers + 1),
        "std": torch.ones(num_layers + 1),
        "num_positions": SEQ_LEN,
        "llm": model.name_or_path,
    }
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "ckpt.pt")
        torch.save(ckpt, path)
        scores = key_norm_linear_scores(path)(model, ids)

    with torch.no_grad():
        cache = model(input_ids=ids, use_cache=True).past_key_values
    assert torch.allclose(scores, key_l2_norm(cache)[0], atol=1e-6)


def test_linear_scores_without_position():
    """A key-norms-only checkpoint ("use_position": False) skips the position column."""
    model, ids = tiny_model(), random_ids()
    num_layers = model.config.num_hidden_layers
    linear = nn.Linear(num_layers, 1)
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, -1] = 1.0  # the last layer
        linear.bias.zero_()
    ckpt = {
        "state_dict": linear.state_dict(),
        "mean": torch.zeros(num_layers),
        "std": torch.ones(num_layers),
        "use_position": False,
        "llm": model.name_or_path,
    }
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "ckpt.pt")
        torch.save(ckpt, path)
        scores = key_norm_linear_scores(path)(model, ids)

    with torch.no_grad():
        cache = model(input_ids=ids, use_cache=True).past_key_values
    assert torch.allclose(scores, key_l2_norm(cache)[-1], atol=1e-6)


def test_passkey_full_precision_matches_plain_greedy():
    model, ids = tiny_model(), random_ids(batch=1)
    num_new = 5

    # Plain greedy decoding with a normal cache
    expected_tokens = []
    with torch.no_grad():
        cache = DynamicCache(config=model.config)
        logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[:, -1]
        for _ in range(num_new):
            token = int(logits.argmax(dim=-1))
            expected_tokens.append(token)
            logits = model(input_ids=torch.tensor([[token]]), past_key_values=cache, use_cache=True).logits[:, -1]

    example = {"prompt_token_ids": ids[0].tolist(), "expected": "unused"}
    trials = policy_passkey(
        model, FakeTokenizer(), [example], None, budget=1.0, low_bits=2,
        is_correct=lambda response, expected: response == expected,
        stop_text=None, window=4, num_sinks=2, max_new_tokens=num_new,
    )
    assert trials[0]["response"] == FakeTokenizer().decode(expected_tokens)


def test_passkey_low_bit_runs_and_reports():
    model, ids = tiny_model(), random_ids(batch=1)
    example = {"prompt_token_ids": ids[0].tolist(), "expected": "1 2 3", "seed": 7}
    trials = policy_passkey(
        model, FakeTokenizer(), [example], windowed_attention_scores(4), budget=0.0, low_bits=2,
        is_correct=lambda response, expected: response == expected,
        stop_text=None, window=4, num_sinks=2, max_new_tokens=3,
    )
    trial = trials[0]
    assert "prompt_token_ids" not in trial and trial["seed"] == 7
    assert isinstance(trial["correct"], bool) and len(trial["response"].split()) == 3


def test_position_and_random_scores():
    ids = random_ids()
    position = position_scores()(None, ids)
    assert (position[:, :-1] > position[:, 1:]).all()  # earlier tokens score higher
    assert torch.equal(random_scores(seed=3)(None, ids), random_scores(seed=3)(None, ids))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"passed  {name}")
    print(f"all {len(tests)} tests passed")