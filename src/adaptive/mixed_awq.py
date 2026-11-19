"""
Per-layer mixed-precision AWQ (W4/W8/FP16 per unit).

vLLM's AWQ format has one bit-width for the whole model and its GPTQ path is
symmetric-only, so a mixed AWQ model is served as a compressed-tensors
pack-quantized checkpoint (WNA16, one Marlin kernel per layer).

1. Base: the same AWQ as the static baseline (GPTQModel, W4 g128 zero-point,
   same calibration blocks), run once with the per-group smoothing scales
   recorded. Shared by every allocation.
2. Per allocation, the base is converted:
   - 4-bit units: the AWQ codes, scales and zeros repacked as-is (lossless),
   - 8-bit units: symmetric RTN (GPTQ's grid) of the AWQ-smoothed weight,
     W * s rebuilt from the original weights and the recorded scales, unclipped.
     Marlin serves zero points only at 4 bits (uint4); 8-bit with zero points
     falls back to a JIT kernel, which would also make TPOT incomparable,
   - 16-bit units: the AWQ-smoothed weight in fp16,
   - lm_head (16 or 8): AWQ leaves it untouched; 8-bit uses the same
     symmetric RTN as the mixed GPTQ checkpoints, so both families match there.
   Norms and embeddings come from the base (AWQ folds 1/s into the norms).

An allocation with every unit at 4 bits and lm_head at 16 reproduces the AWQ
baseline exactly, only in the other format (a check on the conversion and on
kernel parity).
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import torch
from huggingface_hub import save_torch_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from src.adaptive.allocation import all_unit_names, build_allocation, summarize, unit_pattern
from src.adaptive.mixed_quantizer import ALLOCATION_FILE, LM_HEAD_BITS, rtn_sym
from src.adaptive.units import UNIT_GROUPS
from src.baselines.awq_quantizer import AWQQuantizer
from src.baselines.base_quantizer import QuantizerFactory
from src.config.settings import GROUP_SIZE, MIXED_AWQ_OUTPUT_BASE
from src.utils.logger import get_logger

logger = get_logger()

BASE_DIR = MIXED_AWQ_OUTPUT_BASE / "base-w4"
SCALES_FILE = "awq_scales.safetensors"
AWQ_TENSORS = (".qweight", ".qzeros", ".scales")
COPY_FILES = ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "chat_template.jinja")


# ── AWQ scale capture ──────────────────────────────────────────────────────

@contextmanager
def record_awq_scales():
    """Collect (prev_op, layers, scales) for every AWQ scaling group, with full module names.

    The AWQ processor prefixes its per-layer scales_list with the layer path through
    append_str_prefix right after applying it; wrapping that call sees every group.
    Clip entries (name, max_val) pass through the same call and are skipped.
    """
    import gptqmodel.looper.awq_processor as awq_processor

    records: List[Dict[str, Any]] = []
    original = awq_processor.append_str_prefix

    def recording(x, prefix):
        out = original(x, prefix)
        for entry in out:
            if isinstance(entry, tuple) and len(entry) == 4:
                prev_op, layers, scales, _loss = entry
                records.append({"prev_op": prev_op, "layers": list(layers),
                                "scales": scales.detach().float().cpu()})
        return out

    awq_processor.append_str_prefix = recording
    try:
        yield records
    finally:
        awq_processor.append_str_prefix = original


def save_scales(records: List[Dict[str, Any]], path: Path):
    tensors = {r["prev_op"]: r["scales"].contiguous() for r in records}
    groups = {r["prev_op"]: r["layers"] for r in records}
    save_file(tensors, str(path), metadata={"groups": json.dumps(groups)})


def load_scales(path: Path) -> List[Dict[str, Any]]:
    with safe_open(str(path), "pt") as f:
        groups = json.loads(f.metadata()["groups"])
    tensors = load_file(str(path))
    return [{"prev_op": p, "layers": layers, "scales": tensors[p]} for p, layers in groups.items()]


# ── Checkpoint access ──────────────────────────────────────────────────────

class TensorStore:
    """Lazy tensor reads from a (possibly sharded) safetensors checkpoint."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            weight_map = json.load(open(index))["weight_map"]
        else:
            with safe_open(str(model_dir / "model.safetensors"), "pt") as f:
                weight_map = {k: "model.safetensors" for k in f.keys()}
        self.paths = {k: model_dir / shard for k, shard in weight_map.items()}

    def keys(self) -> List[str]:
        return list(self.paths)

    def get(self, key: str) -> torch.Tensor:
        with safe_open(str(self.paths[key]), "pt") as f:
            return f.get_tensor(key)


def smoothed_weight(module: str, original: TensorStore, records: List[Dict[str, Any]]) -> torch.Tensor:
    """The module's weight as AWQ scaled it (before clipping), in float32.

    Inputs of every group's layers are multiplied by s; a Linear prev_op has its
    last len(s) output rows divided by s (AWQ's scale_fc_fc). Norm prev_ops are
    read from the base checkpoint, where AWQ already divided them.
    """
    w = original.get(f"{module}.weight").float()
    for r in records:
        s = r["scales"]
        if module in r["layers"]:
            w *= s.view(1, -1)
        if r["prev_op"] == module:
            w[-s.numel():] /= s.view(-1, 1)
    return w


# ── Quantized tensors (compressed-tensors pack-quantized layout) ──────────

def awq_codes(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor, bits: int):
    """Unpack an AWQ GEMM module into codes q [out, in], zeros [out, groups], scales [out, groups]."""
    from gptqmodel.quantization.awq.utils.packing_utils import reverse_awq_order, unpack_awq

    q, z = unpack_awq(qweight, qzeros, bits)
    q, z = reverse_awq_order(q, z, bits)
    mask = 2 ** bits - 1
    q = torch.bitwise_and(q.to(torch.int32), mask)
    z = torch.bitwise_and(z.to(torch.int32), mask)
    return q.t().contiguous(), z.t().contiguous(), scales.t().contiguous()


def ct_tensors(prefix: str, q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor | None,
               bits: int) -> Dict[str, torch.Tensor]:
    """compressed-tensors tensors from unsigned codes: weight_packed [out, in * bits / 32],
    weight_scale [out, groups], weight_zero_point [out * bits / 32, groups] (asymmetric only)."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32

    offset = 2 ** (bits - 1)  # pack_to_int32 takes signed int8 and adds the offset back
    out = {
        f"{prefix}.weight_packed": pack_to_int32((q - offset).to(torch.int8), bits, packed_dim=1),
        f"{prefix}.weight_scale": scale.to(torch.float16).contiguous(),
        f"{prefix}.weight_shape": torch.tensor(list(q.shape), dtype=torch.int64),
    }
    if zero is not None:
        out[f"{prefix}.weight_zero_point"] = pack_to_int32((zero - offset).to(torch.int8), bits, packed_dim=0)
    return out


def ct_quantization_config(allocation: Dict[str, int], group_size: int = GROUP_SIZE) -> Dict[str, Any]:
    """One config group per (bits, symmetric) with explicit per-unit regex targets.

    4-bit decoder units are asymmetric (AWQ), 8-bit units and lm_head symmetric;
    16-bit decoder units are ignored.
    The unit regexes match both unfused and vLLM's fused names.
    """
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationConfig, QuantizationScheme, QuantizationStatus,
    )

    groups = summarize({u: b for u, b in allocation.items() if u != "lm_head"})
    config_groups = {}
    for bits, units in sorted(groups.items()):
        if bits == "16":
            continue
        config_groups[f"group_w{bits}"] = QuantizationScheme(
            targets=[f"re:{unit_pattern(u)}" for u in units],
            weights=QuantizationArgs(num_bits=int(bits), type="int", symmetric=int(bits) != 4,
                                     strategy="group", group_size=group_size),
        )
    if allocation["lm_head"] != 16:
        config_groups["group_lm_head"] = QuantizationScheme(
            targets=["lm_head"],
            weights=QuantizationArgs(num_bits=allocation["lm_head"], type="int", symmetric=True,
                                     strategy="group", group_size=group_size),
        )
    config = QuantizationConfig(
        config_groups=config_groups,
        quantization_status=QuantizationStatus.COMPRESSED,
        format=CompressionFormat.pack_quantized.value,
        ignore=[f"re:{unit_pattern(u)}" for u in groups.get("16", [])],
    )
    out = config.model_dump(mode="json")
    out["quant_method"] = "compressed-tensors"
    return out


@torch.no_grad()
def convert_awq_checkpoint(base_dir: Path, output_dir: Path, allocation: Dict[str, int],
                           original_dir: str | Path, base_bits: int = 4,
                           group_size: int = GROUP_SIZE) -> Path:
    """Write the compressed-tensors checkpoint of `allocation` from the AWQ base."""
    base = TensorStore(base_dir)
    original = TensorStore(original_dir)
    needs_smoothing = any(b != base_bits for u, b in allocation.items() if u != "lm_head")
    records = load_scales(base_dir / SCALES_FILE) if needs_smoothing else []

    # Norms, embeddings and the untouched lm_head
    tensors = {k: base.get(k) for k in base.keys() if not k.endswith(AWQ_TENSORS)}

    for unit in all_unit_names():
        if unit == "lm_head":
            continue
        _, layer, kind = unit.split(".")
        sub, projs = UNIT_GROUPS[kind]
        bits = allocation[unit]
        for proj in projs:
            module = f"model.layers.{layer}.{sub}.{proj}"
            if bits == base_bits:
                q, z, s = awq_codes(base.get(f"{module}.qweight"), base.get(f"{module}.qzeros"),
                                    base.get(f"{module}.scales"), base_bits)
                tensors.update(ct_tensors(module, q, s, z, bits))
            elif bits == 16:
                tensors[f"{module}.weight"] = smoothed_weight(module, original, records).to(torch.float16)
            else:
                q, s = rtn_sym(smoothed_weight(module, original, records), bits, group_size)
                tensors.update(ct_tensors(module, q, s, None, bits))

    if allocation["lm_head"] != 16:
        q, s = rtn_sym(tensors.pop("lm_head.weight"), allocation["lm_head"], group_size)
        tensors.update(ct_tensors("lm_head", q, s, None, allocation["lm_head"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing compressed-tensors checkpoint ({len(tensors)} tensors) to {output_dir}...")
    save_torch_state_dict(tensors, str(output_dir), max_shard_size="5GB")

    with open(base_dir / "config.json") as f:
        cfg = json.load(f)
    cfg["quantization_config"] = ct_quantization_config(allocation, group_size)
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for name in COPY_FILES:
        if (base_dir / name).exists():
            shutil.copy(base_dir / name, output_dir / name)
    return output_dir


def ct_unit_bits(model_path: str | Path) -> Dict[str, int]:
    """Bits stored per unit in a compressed-tensors checkpoint (weight_packed columns vs weight_shape)."""
    store = TensorStore(model_path)
    keys = set(store.keys())

    def module_bits(module: str) -> int:
        if f"{module}.weight_packed" not in keys:
            return 16
        in_features = int(store.get(f"{module}.weight_shape")[1])
        return 32 * store.get(f"{module}.weight_packed").shape[1] // in_features

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


# ── Quantizer ──────────────────────────────────────────────────────────────

@QuantizerFactory.register("mixed-awq")
class MixedAWQQuantizer(AWQQuantizer):
    """AWQ with per-unit bit-widths, served as compressed-tensors."""

    def __init__(self, name: str, assignments: Dict[str, int], meta: Dict[str, Any] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.meta = meta or {}
        self.allocation = build_allocation(assignments, base_bits=self.bits)
        if self.allocation["lm_head"] not in LM_HEAD_BITS:
            raise ValueError(f"lm_head supports {LM_HEAD_BITS} bits, got {self.allocation['lm_head']}")

    @property
    def method_name(self) -> str:
        return "mixed-awq"

    def _get_output_dir(self) -> Path:
        return MIXED_AWQ_OUTPUT_BASE / f"llama3-8b-instruct-{self.name}"

    def quantize(self, calib_data: List[Dict[str, List[int]]]) -> Any:
        """Run the shared AWQ base once (recording scales); later allocations reuse it."""
        if (BASE_DIR / SCALES_FILE).exists():
            logger.info(f"Reusing AWQ base at {BASE_DIR}")
            return None
        with record_awq_scales() as records:
            super().quantize(calib_data)
        logger.info(f"Recorded {len(records)} AWQ scaling groups")
        super().save(BASE_DIR)
        save_scales(records, BASE_DIR / SCALES_FILE)
        self._model = None
        return None

    def save(self, output_dir: str | Path | None = None) -> Path:
        output_dir = Path(output_dir or self._get_output_dir())
        groups = summarize(self.allocation)
        logger.info(f"Converting AWQ base to mixed '{self.name}':")
        for bits, units in sorted(groups.items()):
            shown = units if len(units) <= 12 else f"{len(units)} units"
            logger.info(f"  {bits}-bit: {shown}")
        convert_awq_checkpoint(BASE_DIR, output_dir, self.allocation, self.model_name, self.bits, self.group_size)
        with open(output_dir / ALLOCATION_FILE, "w") as f:
            json.dump({"name": self.name, "base_bits": self.bits, "base": str(BASE_DIR), **self.meta,
                       "allocation": self.allocation}, f, indent=2)

        stored = ct_unit_bits(output_dir)
        mismatched = {u: (b, stored[u]) for u, b in self.allocation.items() if stored[u] != b}
        if mismatched:
            logger.error(f"Checkpoint bits differ from the allocation (wanted, stored): {mismatched}")
        else:
            logger.info(f"Checkpoint bits match the allocation for all {len(stored)} units")
        return output_dir

    def verify(self, model_path: str | Path) -> bool:
        logger.info("mixed-awq checkpoints are compressed-tensors; verify by evaluating with vLLM")
        return True
