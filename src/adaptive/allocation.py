"""
Per-unit bit allocations and their GPTQ `dynamic` config.

An allocation maps every unit (see src.adaptive.units) to a bit-width:
4 or 8 (GPTQ sym, served by gptq_marlin) or 16 (left unquantized).
Decoder units default to the base bits; lm_head defaults to 16, like the static baselines.

The `dynamic` rules are regexes that both GPTQModel (unfused names, at
quantization time) and vLLM (fused qkv_proj / gate_up_proj names, at load
time) match against full module names, e.g. model.layers.1.mlp.down_proj.
"""

import json
import struct
from pathlib import Path
from typing import Dict, List

from src.adaptive.units import UNIT_GROUPS

NUM_LAYERS = 32  # Llama-3-8B

# kind -> regex alternation over the unfused projections and vLLM's fused name
_KIND_PATTERNS = {
    "qkv": r"self_attn\.(q|k|v|qkv)_proj",
    "o": r"self_attn\.o_proj",
    "gate_up": r"mlp\.(gate|up|gate_up)_proj",
    "down": r"mlp\.down_proj",
}


def all_unit_names(num_layers: int = NUM_LAYERS) -> List[str]:
    return [f"layers.{i}.{kind}" for i in range(num_layers) for kind in UNIT_GROUPS] + ["lm_head"]


def unit_pattern(unit: str) -> str:
    """Regex matching every module of `unit` under both unfused and fused names."""
    if unit == "lm_head":
        return r"lm_head$"
    _, layer, kind = unit.split(".")
    return rf"model\.layers\.{layer}\.{_KIND_PATTERNS[kind]}$"


def build_allocation(overrides: Dict[str, int], base_bits: int = 4, num_layers: int = NUM_LAYERS) -> Dict[str, int]:
    """Full unit -> bits map: base bits everywhere, lm_head at 16, then `overrides`."""
    allocation = {name: base_bits for name in all_unit_names(num_layers)}
    allocation["lm_head"] = 16
    for unit, bits in overrides.items():
        if unit not in allocation:
            raise ValueError(f"Unknown unit: {unit}")
        if bits not in (4, 8, 16):
            raise ValueError(f"{unit}: bits must be 4, 8 or 16 (vLLM gptq_marlin), got {bits}")
        allocation[unit] = bits
    return allocation


def parse_assignments(items: List[str]) -> Dict[str, int]:
    """CLI form ["lm_head=8", "layers.1.down=8"] -> {"lm_head": 8, "layers.1.down": 8}."""
    out = {}
    for item in items or []:
        unit, _, bits = item.partition("=")
        out[unit] = int(bits)
    return out


def dynamic_config(allocation: Dict[str, int], base_bits: int = 4) -> Dict[str, Dict]:
    """GPTQ `dynamic` rules for every unit that differs from the base bits.

    "+:" rules override bits; "-:" rules leave the unit unquantized (FP16).
    lm_head is controlled by the `lm_head` config flag, not a "-:" rule.
    """
    dynamic = {}
    for unit, bits in allocation.items():
        if bits == base_bits or (unit == "lm_head" and bits == 16):
            continue
        if bits == 16:
            dynamic[f"-:{unit_pattern(unit)}"] = {}
        else:
            dynamic[f"+:{unit_pattern(unit)}"] = {"bits": bits}
    return dynamic


def checkpoint_unit_bits(model_path: str | Path) -> Dict[str, int]:
    """Bits actually stored per unit, read from the safetensors headers.

    GPTQ packs `qweight` as [in_features * bits / 32, out_features] int32 and
    stores `g_idx` as [in_features], so bits = 32 * rows / in_features.
    Unquantized modules have a plain `.weight` (16).
    """
    shapes = {}
    for shard in sorted(Path(model_path).glob("*.safetensors")):
        with open(shard, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        shapes.update({k: v["shape"] for k, v in header.items() if k != "__metadata__"})

    def module_bits(module: str) -> int:
        if f"{module}.qweight" in shapes:
            return 32 * shapes[f"{module}.qweight"][0] // shapes[f"{module}.g_idx"][0]
        return 16

    bits = {}
    for unit in all_unit_names():
        if unit == "lm_head":
            modules = ["lm_head"]
        else:
            _, layer, kind = unit.split(".")
            sub, projs = UNIT_GROUPS[kind]
            modules = [f"model.layers.{layer}.{sub}.{p}" for p in projs]
        per_module = {module_bits(m) for m in modules}
        if len(per_module) != 1:
            raise ValueError(f"{unit}: modules stored at mixed bits {per_module}; vLLM requires one precision")
        bits[unit] = per_module.pop()
    return bits


def summarize(allocation: Dict[str, int]) -> Dict[str, List[str]]:
    """bits -> units, omitting the (usually huge) base-bit group."""
    groups: Dict[str, List[str]] = {}
    for unit, bits in allocation.items():
        groups.setdefault(str(bits), []).append(unit)
    return groups
