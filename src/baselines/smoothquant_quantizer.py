"""
SmoothQuant (Activation smoothing + weight quantization) implementation.

W8A8 quantization following Xiao et al. (2023):
1. Collect per-input-channel activation maxima max|X_j| on calibration data.
2. Per channel, s_j = max|X_j|^alpha / max|W_j|^(1-alpha). Fold 1/s into the
   preceding RMSNorm weight and s into the following Linear weights, so the
   FP16 model is mathematically unchanged but activation outliers move into weights.
   Llama mappings: input_layernorm -> q/k/v_proj, post_attention_layernorm -> gate/up_proj.
3. Quantize every decoder Linear weight to INT8 (symmetric, per output channel).
   Activations are quantized at runtime to INT8 (symmetric, dynamic per token).

The model is saved as a compressed-tensors "int-quantized" checkpoint, which
vLLM serves with its INT8 W8A8 CUTLASS kernels. lm_head and embeddings stay FP16.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from huggingface_hub import save_torch_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines.base_quantizer import BaseQuantizer, QuantizerFactory
from src.utils.logger import get_logger
from src.config.settings import SMOOTHQUANT_OUTPUT_BASE, SMOOTHQUANT_ALPHA

logger = get_logger()

INT8_MAX = 127.0


def w8a8_quantization_config() -> Dict[str, Any]:
    """compressed-tensors config: INT8 per-channel weights, INT8 dynamic per-token activations."""
    from compressed_tensors import CompressionFormat
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationConfig, QuantizationScheme, QuantizationStatus,
    )

    weights = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="channel", dynamic=False)
    activations = QuantizationArgs(num_bits=8, type="int", symmetric=True, strategy="token", dynamic=True)
    config = QuantizationConfig(
        config_groups={"group_0": QuantizationScheme(
            targets=["Linear"], weights=weights, input_activations=activations,
        )},
        quantization_status=QuantizationStatus.COMPRESSED,
        format=CompressionFormat.int_quantized.value,
        ignore=["lm_head"],
    )
    out = config.model_dump(mode="json")
    out["quant_method"] = "compressed-tensors"
    return out


@QuantizerFactory.register("smoothquant")
class SmoothQuantQuantizer(BaseQuantizer):
    """SmoothQuant quantizer (W8A8) producing a vLLM-servable INT8 checkpoint."""

    alpha: float = SMOOTHQUANT_ALPHA

    @property
    def method_name(self) -> str:
        return "smoothquant"

    def _get_output_dir(self) -> Path:
        return SMOOTHQUANT_OUTPUT_BASE / f"llama3-8b-instruct-w8a8"

    def quantize(self, calib_data: List[Dict[str, List[int]]]) -> Any:
        """Smooth activations into weights, then quantize weights to INT8."""
        logger.info(f"Loading {self.model_name} for SmoothQuant (W8A8, alpha={self.alpha})...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="cuda:0",
        )
        self._model.eval()

        logger.info(f"Collecting per-channel activation maxima on {len(calib_data)} calibration blocks...")
        act_max = self._collect_act_max(calib_data)

        logger.info("Folding smoothing factors into RMSNorm and Linear weights...")
        self._apply_smoothing(act_max)

        logger.info("Quantizing decoder Linear weights to INT8 (per output channel)...")
        self._qweights = self._quantize_weights()

        return self._model

    @torch.no_grad()
    def _collect_act_max(self, calib_data: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Per-input-channel max|X| at the inputs of q_proj (attn) and gate_proj (mlp) of each layer."""
        act_max: Dict[str, torch.Tensor] = {}

        def make_hook(key):
            def hook(module, args):
                x = args[0].detach().abs().reshape(-1, args[0].shape[-1]).amax(dim=0).float()
                act_max[key] = torch.maximum(act_max[key], x) if key in act_max else x
            return hook

        hooks = []
        for i, layer in enumerate(self._model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(make_hook(f"{i}.attn")))
            hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(make_hook(f"{i}.mlp")))

        device = next(self._model.parameters()).device
        for sample in calib_data:
            input_ids = torch.tensor([sample["input_ids"]], device=device)
            self._model.model(input_ids=input_ids)  # decoder only: skip lm_head logits

        for h in hooks:
            h.remove()
        return act_max

    @torch.no_grad()
    def _apply_smoothing(self, act_max: Dict[str, torch.Tensor]):
        """s_j = max|X_j|^a / max|W_j|^(1-a); norm.weight /= s, linear.weight[:, j] *= s_j."""
        for i, layer in enumerate(self._model.model.layers):
            attn, mlp = layer.self_attn, layer.mlp
            mappings = [
                (layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj], f"{i}.attn"),
                (layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj], f"{i}.mlp"),
            ]
            for norm, linears, key in mappings:
                x_max = act_max[key].clamp(min=1e-5)
                w_max = torch.stack(
                    [lin.weight.abs().amax(dim=0).float() for lin in linears]
                ).amax(dim=0).clamp(min=1e-5)
                s = (x_max.pow(self.alpha) / w_max.pow(1 - self.alpha)).clamp(min=1e-5)

                norm.weight.copy_((norm.weight.float() / s).to(norm.weight.dtype))
                for lin in linears:
                    lin.weight.copy_((lin.weight.float() * s.unsqueeze(0)).to(lin.weight.dtype))

    @torch.no_grad()
    def _quantize_weights(self) -> Dict[str, tuple]:
        """Symmetric per-output-channel INT8: scale = max|W_i| / 127."""
        qweights = {}
        for name, module in self._model.model.layers.named_modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight.float()
                scale = (w.abs().amax(dim=1, keepdim=True) / INT8_MAX).clamp(min=1e-8)
                q = (w / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
                qweights[f"model.layers.{name}"] = (q, scale)
        logger.info(f"Quantized {len(qweights)} Linear layers to INT8.")
        return qweights

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Save as a compressed-tensors int-quantized checkpoint."""
        if output_dir is None:
            output_dir = self._get_output_dir()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving SmoothQuant W8A8 model to {output_dir}...")
        # Tensors stay on GPU; safetensors copies one small shard at a time to host
        # memory (the Slurm job has ~24 GB RAM, less than a full CPU state dict).
        state = {k: v.detach() for k, v in self._model.state_dict().items()}
        for name, (q, scale) in self._qweights.items():
            state[f"{name}.weight"] = q.contiguous()
            state[f"{name}.weight_scale"] = scale.contiguous()
        save_torch_state_dict(state, str(output_dir), max_shard_size="2GB")

        self._model.config.quantization_config = w8a8_quantization_config()
        self._model.config.save_pretrained(str(output_dir))
        self._model.generation_config.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

        with open(output_dir / "smoothquant_meta.json", "w") as f:
            json.dump({"smoothing_alpha": self.alpha,
                       "mappings": ["input_layernorm->q/k/v_proj", "post_attention_layernorm->gate/up_proj"],
                       "weights": "int8 sym per-channel", "activations": "int8 sym dynamic per-token"},
                      f, indent=2)

        logger.info(f"Done! Model saved to {output_dir}")
        return output_dir

    def verify(self, model_path: str | Path) -> bool:
        """Verify the W8A8 checkpoint loads (HF fake-quant path) and generates correctly."""
        from src.utils.model_loader import ModelLoader

        logger.info(f"Verifying W8A8 model at {model_path}...")
        try:
            model, tokenizer = ModelLoader("smoothquant", bits=8).load(str(model_path))
            device = next(model.parameters()).device

            prompt = "Explain quantization in one sentence: "
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=30, do_sample=False)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Verification output: {result}")
            logger.info("Verification passed!")
            return True
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False
