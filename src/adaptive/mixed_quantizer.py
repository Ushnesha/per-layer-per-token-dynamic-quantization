"""
Per-layer mixed-precision GPTQ (W4/W8/FP16 per unit).

Runs the same GPTQ as the static baseline (sym, g128, no act-order, same
calibration blocks) with a `dynamic` config that overrides the bits of
selected decoder units. The checkpoint is a plain GPTQ checkpoint that vLLM
serves with gptq_marlin, one kernel per layer.

lm_head (16 or 8 bits) is quantized after GPTQ with round-to-nearest on the
same symmetric grid and packed into the GPTQ v1 format. GPTQModel 7.3.4's own
lm_head path fails with offload_to_disk (model.norm is still a meta tensor),
and 8-bit RTN lm_head costs ~1e-4 KL in the sensitivity profile.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from gptqmodel import GPTQModel, GPTQConfig, BACKEND
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from src.adaptive.allocation import (
    build_allocation, checkpoint_unit_bits, dynamic_config, summarize, unit_pattern,
)
from src.baselines.base_quantizer import QuantizerFactory
from src.baselines.gptq_quantizer import GPTQQuantizer
from src.config.settings import GPTQ_DAMP_PERCENT, GROUP_SIZE, MIXED_OUTPUT_BASE
from src.utils.logger import get_logger

logger = get_logger()

ALLOCATION_FILE = "allocation.json"
LM_HEAD_BITS = (8, 16)


@torch.no_grad()
def pack_gptq_rtn(w: torch.Tensor, bits: int, group_size: int = GROUP_SIZE) -> Dict[str, torch.Tensor]:
    """Round-to-nearest symmetric quantization of w [out, in] into GPTQ v1 tensors.

    Same grid as GPTQ sym (scale = 2 * max|w| / maxq, zero = 2^(bits-1)).
    v1 layout: qweight [in / pack, out] packed along the input dim,
    qzeros [groups, out / pack] storing zero - 1, scales [groups, out], g_idx [in].
    """
    out_features, in_features = w.shape
    maxq = 2 ** bits - 1
    zero = (maxq + 1) // 2
    pack = 32 // bits

    wg = w.float().reshape(out_features, in_features // group_size, group_size)
    scale = (wg.abs().amax(dim=-1, keepdim=True) * 2 / maxq).clamp(min=1e-8)
    q = (wg / scale).round().add(zero).clamp(0, maxq).to(torch.int32).reshape(out_features, in_features)

    q = q.t().contiguous().reshape(in_features // pack, pack, out_features)
    qweight = torch.zeros(in_features // pack, out_features, dtype=torch.int32)
    for j in range(pack):
        qweight |= q[:, j] << (bits * j)

    zero_word = 0
    for j in range(pack):
        zero_word |= (zero - 1) << (bits * j)
    zero_word = torch.tensor(zero_word, dtype=torch.int64).to(torch.int32)  # wrap to signed int32
    num_groups = in_features // group_size
    qzeros = zero_word.expand(num_groups, out_features // pack).contiguous()

    return {
        "qweight": qweight,
        "qzeros": qzeros,
        "scales": scale.squeeze(-1).t().contiguous().to(torch.float16),
        "g_idx": (torch.arange(in_features) // group_size).to(torch.int32),
    }


def quantize_lm_head_in_checkpoint(model_dir: Path, bits: int, group_size: int = GROUP_SIZE):
    """Replace lm_head.weight in a saved GPTQ checkpoint by RTN GPTQ tensors and mark lm_head quantized."""
    index_path = model_dir / "model.safetensors.index.json"
    index = json.load(open(index_path)) if index_path.exists() else None
    shard = model_dir / (index["weight_map"]["lm_head.weight"] if index else "model.safetensors")

    tensors = load_file(str(shard))
    weight = tensors.pop("lm_head.weight")
    packed = pack_gptq_rtn(weight, bits, group_size)
    tensors.update({f"lm_head.{k}": v for k, v in packed.items()})
    save_file(tensors, str(shard), metadata={"format": "pt"})

    if index:
        del index["weight_map"]["lm_head.weight"]
        index["weight_map"].update({f"lm_head.{k}": shard.name for k in packed})
        if "total_size" in index.get("metadata", {}):
            index["metadata"]["total_size"] += (sum(v.numel() * v.element_size() for v in packed.values())
                                                - weight.numel() * weight.element_size())
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    # vLLM reads quantization_config from config.json (or quantize_config.json)
    rule = {f"+:{unit_pattern('lm_head')}": {"bits": bits}}
    for name in ("config.json", "quantize_config.json"):
        path = model_dir / name
        if not path.exists():
            continue
        with open(path) as f:
            cfg = json.load(f)
        qcfg = cfg["quantization_config"] if name == "config.json" else cfg
        qcfg["lm_head"] = True
        if bits != qcfg["bits"]:
            qcfg["dynamic"] = {**(qcfg.get("dynamic") or {}), **rule}
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
    logger.info(f"lm_head quantized to {bits}-bit (RTN, g{group_size}) in {shard.name}")


@QuantizerFactory.register("mixed")
class MixedGPTQQuantizer(GPTQQuantizer):
    """GPTQ with per-unit bit-widths."""

    def __init__(self, name: str, assignments: Dict[str, int], meta: Dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.meta = meta or {}  # e.g. budget and the allocator's predicted size / cost
        self.allocation = build_allocation(assignments, base_bits=self.bits)
        if self.allocation["lm_head"] not in LM_HEAD_BITS:
            raise ValueError(f"lm_head supports {LM_HEAD_BITS} bits, got {self.allocation['lm_head']}")
        self.dynamic = dynamic_config(self.allocation, base_bits=self.bits)
        # GPTQ only sees decoder units; lm_head is handled after saving
        self.gptq_dynamic = {k: v for k, v in self.dynamic.items() if "lm_head" not in k}

    @property
    def method_name(self) -> str:
        return "mixed"

    def _get_output_dir(self) -> Path:
        return MIXED_OUTPUT_BASE / f"llama3-8b-instruct-{self.name}"

    def quantize(self, calib_data: List[Dict[str, List[int]]]) -> Any:
        groups = summarize(self.allocation)
        logger.info(f"Loading {self.model_name} for mixed GPTQ '{self.name}' (base W{self.bits}A16)...")
        for bits, units in sorted(groups.items()):
            shown = units if len(units) <= 12 else f"{len(units)} units"
            logger.info(f"  {bits}-bit: {shown}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        quantize_config = GPTQConfig(
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            desc_act=False,
            act_group_aware=False,
            damp_percent=GPTQ_DAMP_PERCENT,
            dynamic=self.gptq_dynamic or None,
        )
        self._model = GPTQModel.from_pretrained(self.model_name, quantize_config=quantize_config)

        logger.info(f"Quantizing with {len(calib_data)} calibration blocks...")
        self._model.quantize(
            calib_data,
            tokenizer=self._tokenizer,
            batch_size=1,
            backend=BACKEND.GPTQ_TORCH,  # avoid JIT-built CUDA kernels; checkpoint format is unchanged
        )
        return self._model

    def save(self, output_dir: str | Path | None = None) -> Path:
        output_dir = super().save(output_dir)
        if self.allocation["lm_head"] != 16:
            quantize_lm_head_in_checkpoint(output_dir, self.allocation["lm_head"], self.group_size)
        with open(output_dir / ALLOCATION_FILE, "w") as f:
            json.dump({"name": self.name, "base_bits": self.bits, **self.meta,
                       "allocation": self.allocation, "dynamic": self.dynamic}, f, indent=2)

        stored = checkpoint_unit_bits(output_dir)
        mismatched = {u: (b, stored[u]) for u, b in self.allocation.items() if stored[u] != b}
        if mismatched:
            logger.error(f"Checkpoint bits differ from the allocation (wanted, stored): {mismatched}")
        else:
            logger.info(f"Checkpoint bits match the allocation for all {len(stored)} units")
        return output_dir
