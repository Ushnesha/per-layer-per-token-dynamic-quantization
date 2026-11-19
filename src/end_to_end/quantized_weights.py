"""Load the team's weight checkpoints as plain Hugging Face models, without gptqmodel.

Ushnesha's static and per-layer mixed checkpoints are GPTQ (v1) files built for vLLM's GPTQ
kernels, and her Hugging Face loader needs the gptqmodel package. The end-to-end pipeline only
needs the model to compute what the quantized model computes, so every quantized layer is
unpacked back to FP16 once, at load time. These are fake-quantized weights, the same idea as v0
of the KV cache: the outputs are the quantized model's, but the weights take FP16 memory on the
GPU. Report weight memory from the checkpoint size (her vLLM numbers), not from this model.

Supported: the plain FP16 model and GPTQ-format checkpoints (models/gptq/..., and
models/mixed/llama3-...). The AWQ-format ones (models/awq/..., models/mixed/awq/...) pack their
weights differently and raise an error.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

GPTQ_PARTS = ("qweight", "qzeros", "scales", "g_idx")


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


@torch.no_grad()
def _copy_dequantized(weight: torch.Tensor, parts, bits: int, chunk: int = 4096) -> None:
    """Unpacks one GPTQ layer into `weight` [out, in], a slice of outputs at a time to keep memory low."""
    qweight, qzeros, scales, g_idx = parts
    pack = 32 // bits
    device = weight.device
    g_idx = g_idx.to(device)
    for start in range(0, qweight.shape[1], chunk):
        end = min(start + chunk, qweight.shape[1])
        weight[start:end].copy_(
            dequantize_gptq(
                qweight[:, start:end].to(device),
                qzeros[:, start // pack : end // pack].to(device),
                scales[:, start:end].to(device),
                g_idx,
                bits,
            )
        )


@torch.no_grad()
def load_model(path: str | Path, dtype: torch.dtype = torch.float16, device: str = "cuda"):
    """Returns (model, tokenizer) for an FP16 or GPTQ checkpoint, as a plain HF model on `device`."""
    path = Path(path)
    config = AutoConfig.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    qcfg = getattr(config, "quantization_config", None)
    if qcfg is None:  # the plain FP16 model
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
        return model, tokenizer

    qcfg = qcfg if isinstance(qcfg, dict) else qcfg.to_dict()
    method, fmt = qcfg.get("quant_method"), qcfg.get("checkpoint_format", "gptq")
    if method != "gptq" or fmt != "gptq":
        raise NotImplementedError(
            f"{path}: only GPTQ v1 checkpoints are supported (quant_method={method}, checkpoint_format={fmt})"
        )
    desc_act = bool(qcfg.get("desc_act", False))
    group_size = int(qcfg["group_size"])

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
    with ExitStack() as stack:
        files = {shard: stack.enter_context(safe_open(str(path / shard), framework="pt")) for shard in set(weight_map.values())}

        def get(name):
            return files[weight_map[name]].get_tensor(name)

        # Quantized layers: bits = 32 * qweight rows / in_features (the same rule as allocation.checkpoint_unit_bits)
        reordered = 0
        for module in sorted({name[: -len(".qweight")] for name in weight_map if name.endswith(".qweight")}):
            parts = [get(f"{module}.{part}") for part in GPTQ_PARTS]
            in_features = parts[3].numel()
            bits = 32 * parts[0].shape[0] // in_features
            if not desc_act:
                # Without act-order, vLLM ignores g_idx and puts the columns in groups in order.
                # Do the same, because some checkpoints store a g_idx that vLLM never reads.
                sequential = torch.arange(in_features, dtype=torch.int32) // group_size
                reordered += int(not torch.equal(parts[3].to(torch.int32), sequential))
                parts[3] = sequential
            _copy_dequantized(params[f"{module}.weight"].data, parts, bits)
            loaded.add(f"{module}.weight")
        if reordered:
            print(f"{path}: ignored a non-sequential g_idx in {reordered} layers (desc_act=False, as vLLM does)")

        # Everything stored in full precision: embeddings, norms, unquantized layers
        for name in weight_map:
            if name.rsplit(".", 1)[-1] in GPTQ_PARTS or name not in params:
                continue
            params[name].data.copy_(get(name))
            loaded.add(name)

    missing = sorted(set(params) - loaded)
    if missing:
        raise ValueError(f"{path}: no weights found for {len(missing)} parameters, e.g. {missing[:3]}")
    return model, tokenizer