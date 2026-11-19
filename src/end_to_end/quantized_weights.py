"""Load the team's weight checkpoints as plain Hugging Face models, without gptqmodel.

Ushnesha's checkpoints are built for vLLM's kernels, and her Hugging Face loader needs the
gptqmodel package. The end-to-end pipeline only needs the model to compute what the quantized
model computes, so every quantized layer is unpacked back to FP16 once, at load time. These are
fake-quantized weights, the same idea as v0 of the KV cache: the outputs are the quantized
model's, but the weights take FP16 memory on the GPU. Report weight memory from the checkpoint
size (her vLLM numbers), not from this model.

Supported formats (read from quantization_config in config.json):
    - none: the plain FP16 model (models/llama-3-8b-instruct)
    - GPTQ v1: models/gptq/..., models/mixed/llama3-... (static and per-layer mixed GPTQ)
    - AutoAWQ GEMM, 4-bit: models/awq/... (static AWQ)
    - compressed-tensors pack-quantized: models/mixed/awq/... (per-layer mixed AWQ, where each
      layer is 4-bit AWQ with zero points, 8-bit symmetric, or FP16)
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

GPTQ_PARTS = ("qweight", "qzeros", "scales", "g_idx")
AWQ_PARTS = ("qweight", "qzeros", "scales")
CT_PARTS = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")

# AWQ stores the eight 4-bit codes of each int32 interleaved: the code for output column k of a
# group of eight sits at bits 4 * AWQ_REVERSE_ORDER[k] (the same tables as AutoAWQ and gptqmodel).
AWQ_REVERSE_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


# ── GPTQ v1 ────────────────────────────────────────────────────────────────

@torch.no_grad()
def dequantize_gptq(qweight, qzeros, scales, g_idx, bits: int) -> torch.Tensor:
    """GPTQ v1 tensors -> FP16 weight [out, in]. The same as dequantize_gptq in src/adaptive/mixed_quantizer.py."""
    pack = 32 // bits
    mask = 2**bits - 1
    shifts = torch.arange(pack, device=qweight.device, dtype=torch.int32) * bits
    q = ((qweight.unsqueeze(1) >> shifts.view(1, -1, 1)) & mask).reshape(-1, qweight.shape[1])  # [in, out]
    z = ((qzeros.unsqueeze(2) >> shifts.view(1, 1, -1)) & mask).reshape(qzeros.shape[0], -1) + 1  # [groups, out]
    g = g_idx.long()
    return ((q.float() - z.float()[g]) * scales.float()[g]).t().contiguous().to(torch.float16)


# ── AutoAWQ GEMM ───────────────────────────────────────────────────────────

def _unpack_awq(packed: torch.Tensor) -> torch.Tensor:
    """[rows, cols / 8] int32 in AWQ's interleaved order -> [rows, cols] codes in column order."""
    shifts = 4 * torch.tensor(AWQ_REVERSE_ORDER, dtype=torch.int32, device=packed.device)
    return ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(packed.shape[0], -1)


@torch.no_grad()
def dequantize_awq(qweight, qzeros, scales) -> torch.Tensor:
    """AutoAWQ GEMM 4-bit tensors -> FP16 weight [out, in].

    qweight [in, out / 8] and qzeros [groups, out / 8] pack eight output columns per int32;
    scales is [groups, out]. Each weight is (code - zero) * scale, with no +1 on the zeros (that's
    GPTQ v1 only).
    """
    q = _unpack_awq(qweight)  # [in, out]
    z = _unpack_awq(qzeros)  # [groups, out]
    group_size = qweight.shape[0] // scales.shape[0]
    g = torch.arange(qweight.shape[0], device=qweight.device) // group_size
    return ((q.float() - z.float()[g]) * scales.float()[g]).t().contiguous().to(torch.float16)


# ── compressed-tensors pack-quantized ──────────────────────────────────────

def _unpack_dense(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """compressed-tensors packing along the last dimension: element i of a row sits at bits
    i * bits .. i * bits + bits - 1 of the row's int32s. For 4 and 8 bits, no element crosses a word."""
    per_word = 32 // bits
    shifts = torch.arange(per_word, dtype=torch.int32, device=packed.device) * bits
    return ((packed.unsqueeze(-1) >> shifts) & (2**bits - 1)).reshape(*packed.shape[:-1], -1)


@torch.no_grad()
def dequantize_ct(weight_packed, weight_scale, weight_zero_point, in_features: int, bits: int) -> torch.Tensor:
    """compressed-tensors pack-quantized tensors -> FP16 weight [out, in].

    weight_packed [out, in * bits / 32] holds unsigned codes. weight_scale is [out, groups].
    weight_zero_point [out * bits / 32, groups] holds unsigned zeros packed along the output
    dimension (asymmetric layers only). Symmetric layers have no zero point and are centered at
    2^(bits - 1). Each weight is (code - zero) * scale.
    """
    num_out, num_groups = weight_scale.shape
    q = _unpack_dense(weight_packed, bits)[:, :in_features]  # [out, in]
    if weight_zero_point is None:
        z = torch.full((num_out, num_groups), 2 ** (bits - 1), dtype=torch.float32, device=q.device)
    else:
        z = _unpack_dense(weight_zero_point.t(), bits)[:, :num_out].t().float()  # [out, groups]
    g = torch.arange(in_features, device=q.device) // (in_features // num_groups)
    return ((q.float() - z[:, g]) * weight_scale.float()[:, g]).to(torch.float16)


# ── Loading ────────────────────────────────────────────────────────────────

def _quant_format(path: Path) -> str | None:
    """'gptq', 'awq' or 'compressed-tensors' from config.json, or None for an unquantized model."""
    qcfg = json.loads((path / "config.json").read_text()).get("quantization_config")
    if qcfg is None:
        return None
    method = qcfg.get("quant_method")
    if method == "gptq":
        fmt = qcfg.get("checkpoint_format", "gptq")
        if fmt != "gptq":
            raise NotImplementedError(f"{path}: GPTQ checkpoint_format={fmt} isn't supported, only v1 ('gptq')")
        return "gptq"
    if method == "awq":
        fmt = str(qcfg.get("checkpoint_format") or qcfg.get("format") or qcfg.get("version") or "gemm").lower()
        if fmt != "gemm":
            raise NotImplementedError(f"{path}: AWQ format '{fmt}' isn't supported, only 'gemm'")
        return "awq"
    if method == "compressed-tensors":
        fmt = qcfg.get("format")
        if fmt != "pack-quantized":
            raise NotImplementedError(f"{path}: compressed-tensors format '{fmt}' isn't supported, only 'pack-quantized'")
        return "compressed-tensors"
    raise NotImplementedError(f"{path}: quant_method '{method}' isn't supported")


@torch.no_grad()
def load_model(path: str | Path, dtype: torch.dtype = torch.float16, device: str = "cuda", chunk: int = 4096):
    """Returns (model, tokenizer) for any supported checkpoint, as a plain HF model on `device`.

    Quantized layers are unpacked `chunk` output rows at a time, to keep the extra memory low.
    """
    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    fmt = _quant_format(path)
    if fmt is None:  # the plain FP16 model
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        return model, tokenizer
    qcfg = json.loads((path / "config.json").read_text())["quantization_config"]

    # Build the plain architecture, then fill in its weights from the checkpoint
    del config.quantization_config
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    model.eval()
    model.config._name_or_path = str(path)
    try:
        model.generation_config = GenerationConfig.from_pretrained(path)
    except OSError:
        pass

    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(str(path / "model.safetensors"), framework="pt") as f:
            weight_map = {name: "model.safetensors" for name in f.keys()}

    params = dict(model.named_parameters())
    loaded = set()
    bits_seen: dict[int, int] = {}
    with ExitStack() as stack:
        files = {shard: stack.enter_context(safe_open(str(path / shard), framework="pt")) for shard in set(weight_map.values())}

        def get(name):
            return files[weight_map[name]].get_tensor(name)

        def modules_with(suffix):
            return sorted({name[: -len(suffix)] for name in weight_map if name.endswith(suffix)})

        if fmt == "gptq":
            desc_act = bool(qcfg.get("desc_act", False))
            group_size = int(qcfg["group_size"])
            reordered = 0
            for module in modules_with(".qweight"):
                qweight, qzeros, scales, g_idx = [get(f"{module}.{part}") for part in GPTQ_PARTS]
                in_features = g_idx.numel()
                # bits = 32 * qweight rows / in_features (the same rule as allocation.checkpoint_unit_bits)
                bits = 32 * qweight.shape[0] // in_features
                if not desc_act:
                    # Without act-order, vLLM ignores g_idx and puts the columns in groups in order.
                    # Do the same, because some checkpoints store a g_idx that vLLM never reads.
                    sequential = torch.arange(in_features, dtype=torch.int32) // group_size
                    reordered += int(not torch.equal(g_idx.to(torch.int32), sequential))
                    g_idx = sequential
                weight, pack = params[f"{module}.weight"].data, 32 // bits
                g_idx = g_idx.to(device)
                for start in range(0, weight.shape[0], chunk):
                    end = min(start + chunk, weight.shape[0])
                    weight[start:end].copy_(dequantize_gptq(
                        qweight[:, start:end].to(device), qzeros[:, start // pack : end // pack].to(device),
                        scales[:, start:end].to(device), g_idx, bits,
                    ))
                loaded.add(f"{module}.weight")
                bits_seen[bits] = bits_seen.get(bits, 0) + 1
            if reordered:
                print(f"{path}: ignored a non-sequential g_idx in {reordered} layers (desc_act=False, as vLLM does)")

        elif fmt == "awq":
            for module in modules_with(".qweight"):
                qweight, qzeros, scales = [get(f"{module}.{part}") for part in AWQ_PARTS]
                bits = 32 * qweight.shape[1] // scales.shape[1]
                if bits != 4:
                    raise NotImplementedError(f"{module}: AWQ GEMM with {bits} bits isn't supported, only 4")
                weight = params[f"{module}.weight"].data
                for start in range(0, weight.shape[0], chunk):
                    end = min(start + chunk, weight.shape[0])
                    weight[start:end].copy_(dequantize_awq(
                        qweight[:, start // 8 : end // 8].to(device), qzeros[:, start // 8 : end // 8].to(device),
                        scales[:, start:end].to(device),
                    ))
                loaded.add(f"{module}.weight")
                bits_seen[bits] = bits_seen.get(bits, 0) + 1

        else:  # compressed-tensors
            for module in modules_with(".weight_packed"):
                packed, scale = get(f"{module}.weight_packed"), get(f"{module}.weight_scale")
                zero = get(f"{module}.weight_zero_point") if f"{module}.weight_zero_point" in weight_map else None
                in_features = int(get(f"{module}.weight_shape")[1])
                bits = 32 * packed.shape[1] // in_features
                if bits not in (4, 8):
                    raise NotImplementedError(f"{module}: {bits}-bit compressed-tensors layers aren't supported, only 4 and 8")
                weight, per_word = params[f"{module}.weight"].data, 32 // bits
                for start in range(0, weight.shape[0], chunk):
                    end = min(start + chunk, weight.shape[0])
                    zero_part = None if zero is None else zero[start // per_word : -(-end // per_word)].to(device)
                    weight[start:end].copy_(dequantize_ct(
                        packed[start:end].to(device), scale[start:end].to(device), zero_part, in_features, bits,
                    ))
                loaded.add(f"{module}.weight")
                bits_seen[bits] = bits_seen.get(bits, 0) + 1

        # Everything stored in full precision: embeddings, norms, unquantized (FP16) layers
        for name in weight_map:
            if name in params and name not in loaded:
                params[name].data.copy_(get(name))
                loaded.add(name)

    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"{path}: no weights found for {len(missing)} parameters, e.g. {missing[:3]}")
    layers = ", ".join(f"{count} at {bits}-bit" for bits, count in sorted(bits_seen.items()))
    print(f"{path}: {fmt} checkpoint, unpacked {layers} to FP16")
    return model, tokenizer