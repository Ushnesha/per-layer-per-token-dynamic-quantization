import torch

def sensitivity_scores(kl: torch.Tensor, H: int) -> torch.Tensor:
    """Mean KL over the H positions starting at each quantized token.

    kl is [chunks, j, t]. Returns [chunks, S - H + 1], one score per token that has a full window.
    """
    j_s = torch.arange(kl.shape[-1] - H + 1).unsqueeze(1)  # [num_scored, 1]
    cols = j_s + torch.arange(H)  # [num_scored, H]: positions j .. j + H - 1
    return kl[:, j_s, cols].mean(dim=-1)


def shared_positions(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cuts a and b ([chunks, tokens]) to the token positions both have."""
    cutoff = min(a.shape[1], b.shape[1])
    return a[:, :cutoff], b[:, :cutoff]


def per_chunk_correlation(a: torch.Tensor, b: torch.Tensor, corr_fn) -> torch.Tensor:
    """corr_fn (kendalltau or spearmanr) between a and b, one value per chunk."""
    a, b = shared_positions(a, b)
    a, b = a.numpy(), b.numpy()
    return torch.tensor([corr_fn(a[row], b[row]).statistic for row in range(a.shape[0])])


def per_chunk_top_k_overlap(a: torch.Tensor, b: torch.Tensor, frac: float = 0.1) -> torch.Tensor:
    """Fraction of the top `frac` tokens that a and b share, one value per chunk."""
    a, b = shared_positions(a, b)
    k = int(a.shape[1] * frac)
    top_a = torch.topk(a, k, dim=-1).indices
    top_b = torch.topk(b, k, dim=-1).indices
    return torch.tensor(
        [len(set(top_a[row].tolist()) & set(top_b[row].tolist())) / k for row in range(a.shape[0])]
    )


def compare_windows(scores: dict, pairs: list, metric_fn) -> dict:
    """metric_fn(a, b) for each pair of window sizes, e.g. {"h32_vs_h64": tensor of 20 values}."""
    return {f"h{x}_vs_h{y}": metric_fn(scores[x], scores[y]) for x, y in pairs}


def print_summary(stats: dict, name: str) -> None:
    for comparison, values in stats.items():
        print(f"{comparison} {name}: mean {values.mean():.3f}, min {values.min():.3f}, max {values.max():.3f}")
    print()


def inspect_token(kl, chunks, tokenizer, chunk, position, context_before=13, context_after=8, num_next=13):
    """Prints a token, the text around it, and the KL it causes on the next few predictions."""
    token = tokenizer.convert_ids_to_tokens(chunks[chunk, position].item())
    print(f"chunk {chunk}, position {position}: {token!r}")
    print(f"context: {tokenizer.decode(chunks[chunk, position - context_before : position + context_after])}")

    # The KL at position t is about predicting token t + 1
    kl_values = kl[chunk, position, position : position + num_next].tolist()
    predicted = tokenizer.convert_ids_to_tokens(chunks[chunk, position + 1 : position + 1 + num_next].tolist())
    for t, (next_token, value) in enumerate(zip(predicted, kl_values), start=position):
        print(f"  t = {t}: predicting {next_token!r:<10} KL {value:.2e}")
    print()