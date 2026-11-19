"""
Budget-constrained bit allocation from a sensitivity profile.

Each unit picks one bit-width (decoder units 4 / 8 / 16, lm_head 8 / 16) so
that the summed per-unit cost (reference KL from src.adaptive.sensitivity)
is minimal while the checkpoint's weight storage stays within a budget.
This is a multiple-choice knapsack, solved exactly by DP over 10 KB bins
(each unit's size is rounded up, so at most ~1.3 MB of the budget is lost). Optimal under the additive-cost assumption,
which the evaluated checkpoints test.

Storage is the exact GPTQ v1 byte count per unit (qweight, qzeros, scales,
g_idx), plus the unquantized tensors (embeddings, norms), so predicted sizes
match `weight_storage_gb` in the results.

Usage:
    python -m src.adaptive.allocator --budget 5.5 6.0 6.5
"""

import argparse
import json
import random
import struct
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.adaptive.allocation import build_allocation
from src.config.settings import DEFAULT_MODEL_PATH, GROUP_SIZE, SENSITIVITY_DIR

DEFAULT_PROFILE = SENSITIVITY_DIR / "llama3-8b-instruct.json"
DECODER_BITS = (4, 8, 16)
LM_HEAD_BITS = (8, 16)
BIN_BYTES = 10_000  # DP resolution


def tensor_bytes(model_path: str | Path) -> Dict[str, int]:
    """Byte size of every tensor in a checkpoint, from the safetensors headers."""
    sizes = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        with open(shard, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, info in header.items():
            if name != "__metadata__":
                begin, end = info["data_offsets"]
                sizes[name] = end - begin
    return sizes


def module_bytes(out_features: int, in_features: int, bits: int, group_size: int = GROUP_SIZE) -> int:
    """Stored bytes of one Linear: FP16 weight, or GPTQ v1 qweight + qzeros + scales + g_idx."""
    if bits == 16:
        return out_features * in_features * 2
    groups = in_features // group_size
    return (in_features * out_features * bits // 8      # qweight
            + groups * out_features * bits // 8         # qzeros
            + groups * out_features * 2                 # scales (fp16)
            + in_features * 4)                          # g_idx (int32)


class Allocator:
    """Size model + per-unit costs from a profile; solves allocations for byte budgets."""

    def __init__(self, profile_path: str | Path = DEFAULT_PROFILE, model_path: str = DEFAULT_MODEL_PATH,
                 signal: str = "reference", metric: str = "kl"):
        with open(profile_path) as f:
            profile = json.load(f)
        self.units = profile["units"]
        self.signal, self.metric = signal, metric

        # Module shapes from the FP16 checkpoint; everything outside the units is fixed storage
        fp16 = tensor_bytes(model_path)
        shapes = self._weight_shapes(model_path)
        unit_modules = {m for u in self.units for m in u["modules"]}
        self.fixed_bytes = sum(b for name, b in fp16.items()
                               if name.rsplit(".", 1)[0] not in unit_modules)

        self.options: List[Dict[int, Dict[str, float]]] = []
        for u in self.units:
            bit_choices = LM_HEAD_BITS if u["kind"] == "lm_head" else DECODER_BITS
            opts = {}
            for bits in bit_choices:
                size = sum(module_bytes(*shapes[m], bits) for m in u["modules"])
                opts[bits] = {"bytes": size, "cost": self.cost(u, bits)}
            self.options.append(opts)

    @staticmethod
    def _weight_shapes(model_path: str | Path) -> Dict[str, tuple]:
        shapes = {}
        for shard in sorted(Path(model_path).glob("*.safetensors")):
            with open(shard, "rb") as f:
                header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            for name, info in header.items():
                if name.endswith(".weight") and len(info.get("shape", [])) == 2:
                    shapes[name[: -len(".weight")]] = tuple(info["shape"])
        return shapes

    def cost(self, unit: dict, bits: int) -> float:
        """Profiled cost of `unit` at `bits`; FP16 costs nothing."""
        if bits == 16:
            return 0.0
        scores = unit[self.signal][str(bits)]
        return float(scores[self.metric])

    def size_gb(self, allocation: Dict[str, int]) -> float:
        total = self.fixed_bytes + sum(opts[allocation[u["name"]]]["bytes"]
                                       for u, opts in zip(self.units, self.options))
        return total / 1e9

    def predicted_cost(self, allocation: Dict[str, int]) -> float:
        return sum(opts[allocation[u["name"]]]["cost"] for u, opts in zip(self.units, self.options))

    def solve(self, budget_gb: float) -> Dict[str, int]:
        """Exact minimum-cost allocation with size <= budget_gb (multiple-choice knapsack DP)."""
        capacity = int((budget_gb * 1e9 - self.fixed_bytes) // BIN_BYTES)
        min_bins = sum(min(-(-o["bytes"] // BIN_BYTES) for o in opts.values()) for opts in self.options)
        if capacity < min_bins:
            raise ValueError(f"Budget {budget_gb} GB is below the smallest allocation "
                             f"({(self.fixed_bytes + min_bins * BIN_BYTES) / 1e9:.3f} GB)")

        inf = np.inf
        best = np.full(capacity + 1, inf)
        best[0] = 0.0
        choice = np.zeros((len(self.options), capacity + 1), dtype=np.int8)
        for i, opts in enumerate(self.options):
            new = np.full(capacity + 1, inf)
            for k, (bits, o) in enumerate(opts.items()):
                w = -(-o["bytes"] // BIN_BYTES)  # round up so the budget is never exceeded
                if w > capacity:
                    continue
                cand = np.full(capacity + 1, inf)
                cand[w:] = best[: capacity + 1 - w] + o["cost"]
                better = cand < new
                new[better] = cand[better]
                choice[i, better] = k
            best = new

        # Backtrack from the cheapest reachable capacity
        c = int(np.argmin(best))
        allocation = {}
        for i in range(len(self.options) - 1, -1, -1):
            bits_list = list(self.options[i].keys())
            bits = bits_list[choice[i, c]]
            allocation[self.units[i]["name"]] = bits
            c -= -(-self.options[i][bits]["bytes"] // BIN_BYTES)
        return allocation

    def control_order(self, kind: str, seed: int = 0) -> List[str]:
        """Decoder units in the order a sensitivity-free control upgrades them.

        random    seeded shuffle
        position  outermost layers first (0, 31, 1, 30, ...), all units of a layer together
        """
        names = [u["name"] for u in self.units if u["kind"] != "lm_head"]
        if kind == "random":
            random.Random(seed).shuffle(names)
            return names
        if kind == "position":
            num_layers = max(u["layer"] for u in self.units) + 1
            rank = {}
            for i in range(num_layers):
                rank[i // 2 if i % 2 == 0 else num_layers - 1 - i // 2] = i
            return sorted(names, key=lambda n: rank[int(n.split(".")[1])])
        raise ValueError(f"Unknown control: {kind}")

    def solve_by_order(self, budget_gb: float, order: List[str]) -> Dict[str, int]:
        """Control allocation: lm_head at 8 bits like the guided ones, then decoder
        units upgraded to 8 bits in `order` while they fit the budget."""
        allocation = {u["name"]: (8 if u["kind"] == "lm_head" else 4) for u in self.units}
        if self.size_gb(allocation) > budget_gb:
            raise ValueError(f"Budget {budget_gb} GB is below the smallest control allocation")
        for name in order:
            trial = {**allocation, name: 8}
            if self.size_gb(trial) <= budget_gb:
                allocation = trial
        return allocation


def overrides(allocation: Dict[str, int], base_bits: int = 4) -> Dict[str, int]:
    """Only the units that differ from build_allocation's defaults (base bits, lm_head 16)."""
    defaults = build_allocation({}, base_bits=base_bits)
    return {u: b for u, b in allocation.items() if b != defaults[u]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget-constrained bit allocation")
    parser.add_argument("--budget", type=float, nargs="+", required=True, help="Weight storage budgets (GB)")
    parser.add_argument("--profile", type=str, default=str(DEFAULT_PROFILE))
    parser.add_argument("--signal", type=str, default="reference", choices=["reference", "gptq", "cheap"])
    parser.add_argument("--metric", type=str, default="kl", help="kl | delta_nll (reference, gptq), abs | rel (cheap)")
    args = parser.parse_args()

    alloc = Allocator(args.profile, signal=args.signal, metric=args.metric)
    for budget in args.budget:
        a = alloc.solve(budget)
        upgraded = {u: b for u, b in overrides(a).items()}
        print(f"budget {budget:.2f} GB -> {alloc.size_gb(a):.3f} GB, predicted {args.metric} "
              f"{alloc.predicted_cost(a):.4f}, {len(upgraded)} units changed: {upgraded}")
