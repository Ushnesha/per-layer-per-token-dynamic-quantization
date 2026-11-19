import matplotlib.pyplot as plt
import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _style(ax) -> None:
    ax.set_yscale("log")
    ax.grid(axis="y", which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_sensitivity(scores, window: int, path: str | None = None, zoom: int = 20):
    s = np.clip(_to_numpy(scores), 1e-12, None)  # log axis can't show values <= 0
    positions = np.arange(s.shape[1])
    mean = s.mean(axis=0)
    low, high = np.percentile(s, [25, 75], axis=0)

    fig, (ax_full, ax_zoom) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1]}, sharey=True
    )
    color = SERIES[0]

    ax_full.fill_between(positions, low, high, color=color, alpha=0.18, linewidth=0)
    ax_full.plot(positions, mean, color=color, linewidth=1.5)
    ax_full.set_title("All positions", loc="left", fontsize=10, color=INK)

    z = slice(0, zoom + 1)
    ax_zoom.fill_between(positions[z], low[z], high[z], color=color, alpha=0.18, linewidth=0)
    ax_zoom.plot(positions[z], mean[z], color=color, linewidth=1.5, marker="o", markersize=4)
    ax_zoom.set_xticks(range(0, zoom + 1, 5 if zoom > 10 else 1))
    ax_zoom.set_title(f"First {zoom + 1} positions", loc="left", fontsize=10, color=INK)

    for ax in (ax_full, ax_zoom):
        _style(ax)
        ax.set_xlabel("Quantized token position j", fontsize=9, color=INK_SECONDARY)
    ax_full.set_ylabel(f"Sensitivity\n(mean KL over next {window} positions)", fontsize=9, color=INK_SECONDARY)

    fig.suptitle(
        f"Per-token INT4 sensitivity (mean over {s.shape[0]} chunks, band = 25th–75th percentile)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_sensitivity_by_window(scores_by_window: dict, path: str | None = None):
    if len(scores_by_window) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} window sizes per plot")

    arrays = {h: np.clip(_to_numpy(s), 1e-12, None) for h, s in sorted(scores_by_window.items())}
    common = min(a.shape[1] for a in arrays.values())
    positions = np.arange(common)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for color, (h, a) in zip(SERIES, arrays.items()):
        mean = a[:, :common].mean(axis=0)
        ax.plot(positions, mean, color=color, linewidth=1.5, label=f"H = {h}")
        ax.annotate(
            f"H = {h}",
            xy=(positions[-1], mean[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_SECONDARY,
        )

    _style(ax)
    ax.set_xlabel("Quantized token position j", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Mean sensitivity", fontsize=9, color=INK_SECONDARY)
    ax.set_title(
        f"Sensitivity for different window sizes (positions 0–{common - 1})",
        loc="left",
        fontsize=11,
        color=INK,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig

def _spread_labels(ys, min_gap: float) -> list:
    """Nudges label heights apart so no two are closer than min_gap, keeping their order."""
    order = np.argsort(ys)
    placed = np.array(ys, dtype=np.float64)
    for prev, cur in zip(order[:-1], order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed.tolist()
 
 
def plot_tau_by_layer(tau_by_layer: dict, baselines: dict | None = None, path: str | None = None):
    """Kendall tau against the oracle when each layer's signal is graded on its own.
 
    Args:
        tau_by_layer: {signal name: tau per layer}, up to 3 entries. Each value is either
            [num_layers] (already averaged over chunks) or [num_layers, num_chunks], in which case
            the line is the mean over chunks and the band is the 25th-75th percentile.
        baselines: Optional {name: tau}, drawn as dashed reference lines,
            e.g. {"position": 0.113, "random": 0.003}.
        path: If given, save the figure here (e.g. "results/tau_by_layer.png").
 
    Returns:
        The matplotlib Figure.
    """
    if len(tau_by_layer) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} signals per plot")
    baselines = baselines or {}
 
    fig, ax = plt.subplots(figsize=(9, 4))
    end_labels = []  # (y, text, color) for the labels in the right margin
 
    for color, (name, values) in zip(SERIES, tau_by_layer.items()):
        tau = _to_numpy(values)
        layers = np.arange(tau.shape[0])
        if tau.ndim == 2:
            low, high = np.percentile(tau, [25, 75], axis=1)
            ax.fill_between(layers, low, high, color=color, alpha=0.15, linewidth=0)
            tau = tau.mean(axis=1)
        label = name.replace("_", " ")
        ax.plot(layers, tau, color=color, linewidth=2, marker="o", markersize=4, label=label)
        end_labels.append((tau[-1], label, INK_SECONDARY))
 
    for name, value in baselines.items():
        ax.axhline(value, color=INK_SECONDARY, linewidth=1, linestyle="--", zorder=0)
        end_labels.append((value, f"{name.replace('_', ' ')} (baseline)", INK_SECONDARY))
    ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=0)
 
    # Recessive grid and axes, linear y (tau can be negative)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.set_xticks(range(0, len(layers), 4))
    ax.set_xlim(-0.5, len(layers) - 0.5)
    ax.set_xlabel("Layer", fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("Kendall τ vs. oracle", fontsize=9, color=INK_SECONDARY)
    ax.set_title("How well each layer's signal ranks tokens like the oracle", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="best")
 
    # Direct labels in the right margin, pushed apart so they don't overlap
    y_min, y_max = ax.get_ylim()
    ys = _spread_labels([y for y, _, _ in end_labels], min_gap=0.05 * (y_max - y_min))
    for y, (_, text, color) in zip(ys, end_labels):
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=color, annotation_clip=False)
 
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig