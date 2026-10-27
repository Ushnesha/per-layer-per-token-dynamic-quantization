"""
Unified model loader supporting FP16, AWQ, GPTQ, and SmoothQuant backends.

Usage:
    from src.utils.model_loader import ModelLoader

    # FP16
    loader = ModelLoader("fp16")
    model, tokenizer = loader.load(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    # AWQ-INT4
    loader = ModelLoader("awq", bits=4)
    model, tokenizer = loader.load(model_path="models/awq/llama3-8b-instruct-w4a16")

    # GPTQ-INT4
    loader = ModelLoader("gptq", bits=4)
    model, tokenizer = loader.load(model_path="models/gptq/llama3-8b-instruct-w4a16")

    # SmoothQuant W8A8
    loader = ModelLoader("smoothquant", bits=8)
    model, tokenizer = loader.load(model_path="models/smoothquant/llama3-8b-instruct-w8a8")
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

try:
    from gptqmodel import GPTQModel, BACKEND
    GPTQ_AVAILABLE = True
except ImportError:
    GPTQ_AVAILABLE = False


class W8A8Linear(nn.Module):
    """Fake-quant W8A8 Linear matching the vLLM compressed-tensors INT8 scheme.

    Weights: INT8, symmetric per output channel (`weight`, `weight_scale`).
    Activations: INT8, symmetric, dynamic per token (computed each forward).
    Numerically equivalent to an INT8 GEMM with FP rescaling; used for
    HF-side verification only (benchmarks run on vLLM's real INT8 kernels).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, 1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_scale = (input.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-8)
        x_q = (input.float() / x_scale).round().clamp(-127, 127)
        x_dq = (x_q * x_scale).to(input.dtype)
        w_dq = (self.weight.float() * self.weight_scale).to(input.dtype)
        return nn.functional.linear(x_dq, w_dq, self.bias)


class ModelLoader:
    """Load models for different quantization backends."""

    def __init__(self, method: str, bits: int = 16, device_map: str = "auto"):
        self.method = method.lower()
        self.bits = bits
        self.device_map = device_map

    def load(
        self,
        model_name_or_path: str,
        revision: Optional[str] = None
    ) -> Tuple[torch.nn.Module, AutoTokenizer]:
        loaders = {
            "fp16": self._load_fp16,
            "awq": self._load_awq,
            "gptq": self._load_gptq,
            "smoothquant": self._load_smoothquant,
        }
        loader_fn = loaders.get(self.method)
        if not loader_fn:
            raise ValueError(f"Unknown method: {self.method}. Choose from {list(loaders.keys())}")
        return loader_fn(model_name_or_path, revision)

    def _load_fp16(self, path: str, revision) -> Tuple[torch.nn.Module, AutoTokenizer]:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map=self.device_map,
            revision=revision,
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer

    def _load_awq(self, path: str, revision) -> Tuple[torch.nn.Module, AutoTokenizer]:
        if not GPTQ_AVAILABLE:
            raise ImportError("gptqmodel is not installed. Run: pip install gptqmodel")
        model = GPTQModel.from_quantized(
            path,
            device_map=self.device_map,
            backend=BACKEND.AWQ_TORCH,  # avoid JIT-built CUDA kernels (fail with system GCC 8.5)
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer

    def _load_gptq(self, path: str, revision) -> Tuple[torch.nn.Module, AutoTokenizer]:
        if not GPTQ_AVAILABLE:
            raise ImportError("gptqmodel is not installed. Run: pip install gptqmodel")
        model = GPTQModel.from_quantized(
            path,
            device_map=self.device_map,
            backend=BACKEND.GPTQ_TORCH,  # avoid Marlin JIT build (fails with system GCC 8.5)
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer

    def _load_smoothquant(self, path: str, revision) -> Tuple[torch.nn.Module, AutoTokenizer]:
        """Load the SmoothQuant compressed-tensors W8A8 checkpoint with W8A8Linear layers.

        1. Build the FP16 architecture from config (quantization_config stripped)
        2. Swap decoder Linear layers for W8A8Linear (INT8 weight + per-channel scale)
        3. Load the INT8 / FP16 tensors from the safetensors shards
        """
        from safetensors.torch import load_file
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(path)
        if hasattr(config, "quantization_config"):
            delattr(config, "quantization_config")

        device = "cuda:0" if self.device_map == "auto" else self.device_map
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

        layers_quantized = 0
        for layer in model.model.layers:
            for parent in (layer.self_attn, layer.mlp):
                for child_name, child in list(parent.named_children()):
                    if isinstance(child, nn.Linear):
                        quant = W8A8Linear(child.in_features, child.out_features, child.bias is not None)
                        setattr(parent, child_name, quant.to(device))
                        layers_quantized += 1

        state = {}
        for shard in sorted(Path(path).glob("*.safetensors")):
            state.update(load_file(str(shard), device=device))
        model.load_state_dict(state, strict=True)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"SmoothQuant: loaded {layers_quantized} W8A8 Linear layers")
        return model, tokenizer

    @staticmethod
    def get_device(model: torch.nn.Module) -> torch.device:
        return next(model.parameters()).device


# ── Legacy compatibility ───────────────────────────────────────────────────
_model = None
_tokenizer = None

def get_model_and_tokenizer(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Legacy function for backward compatibility."""
    global _model, _tokenizer
    if _model is None:
        loader = ModelLoader("fp16")
        _model, _tokenizer = loader.load(model_name)
    return _model, _tokenizer