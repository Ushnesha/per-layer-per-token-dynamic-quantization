"""
Allocation units for per-layer mixed precision.

vLLM fuses q/k/v into qkv_proj and gate/up into gate_up_proj, and requires every
shard of a fused layer to share one precision. So the unit of allocation is the
fused group, not the single Linear: 4 units per decoder layer (qkv, o, gate_up,
down) plus lm_head, which vLLM's GPTQ path can also serve quantized.
"""

from dataclasses import dataclass
from typing import List

import torch

from src.config.settings import GROUP_SIZE

# kind -> (submodule, member projections)
UNIT_GROUPS = {
    "qkv": ("self_attn", ["q_proj", "k_proj", "v_proj"]),
    "o": ("self_attn", ["o_proj"]),
    "gate_up": ("mlp", ["gate_proj", "up_proj"]),
    "down": ("mlp", ["down_proj"]),
}


@dataclass
class Unit:
    """A group of Linear layers that must share one precision."""
    name: str           # e.g. "layers.5.gate_up" or "lm_head"
    kind: str           # "qkv" | "o" | "gate_up" | "down" | "lm_head"
    layer: int          # decoder layer index, -1 for lm_head
    modules: List[str]  # HF module names, e.g. "model.layers.5.mlp.gate_proj"
    params: int = 0


def build_units(model, include_lm_head: bool = True) -> List[Unit]:
    """Enumerate allocation units of a Llama-style HF model, with parameter counts."""
    named = dict(model.named_modules())
    units = []
    for i in range(model.config.num_hidden_layers):
        for kind, (sub, projs) in UNIT_GROUPS.items():
            modules = [f"model.layers.{i}.{sub}.{p}" for p in projs]
            params = sum(named[m].weight.numel() for m in modules)
            units.append(Unit(f"layers.{i}.{kind}", kind, i, modules, params))
    if include_lm_head:
        units.append(Unit("lm_head", "lm_head", -1, ["lm_head"], named["lm_head"].weight.numel()))
    return units


@torch.no_grad()
def fake_quantize(w: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Round-to-nearest symmetric group quantization, same grid as GPTQ sym (uint4b8 / uint8b128).

    Returns the dequantized weight in w's dtype. GPTQ's error compensation is not
    applied, so this is a proxy for how much a unit suffers at `bits`, used for ranking.
    """
    out_features, in_features = w.shape
    wg = w.float().reshape(out_features, in_features // group_size, group_size)
    maxq = 2 ** bits - 1
    scale = (wg.abs().amax(dim=-1, keepdim=True) * 2 / maxq).clamp(min=1e-8)
    zero = (maxq + 1) / 2
    q = (wg / scale).round().add(zero).clamp(0, maxq)
    return ((q - zero) * scale).reshape(out_features, in_features).to(w.dtype)
