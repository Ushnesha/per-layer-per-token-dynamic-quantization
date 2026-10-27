"""
AWQ (Activation-Aware Weight Quantization) implementation.

Uses GPTQModel's AWQ implementation for weight-only quantization (W4A16).
Exports the standard AutoAWQ GEMM checkpoint format, which vLLM serves
with the prebuilt awq_marlin kernel.
"""

from pathlib import Path
from typing import Any, Dict, List

from gptqmodel import GPTQModel, AWQConfig, BACKEND
from transformers import AutoTokenizer

from src.baselines.base_quantizer import BaseQuantizer, QuantizerFactory
from src.utils.logger import get_logger
from src.config.settings import AWQ_OUTPUT_BASE

logger = get_logger()


@QuantizerFactory.register("awq")
class AWQQuantizer(BaseQuantizer):
    """AWQ quantizer using GPTQModel library."""

    @property
    def method_name(self) -> str:
        return "awq"

    def _get_output_dir(self) -> Path:
        return AWQ_OUTPUT_BASE / f"llama3-8b-instruct-w{self.bits}a16"

    def quantize(self, calib_data: List[Dict[str, List[int]]]) -> Any:
        """Quantize model using AWQ."""
        logger.info(f"Loading {self.model_name} for AWQ quantization (W{self.bits}A16)...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Standard AWQ config: asymmetric (zero-point), GEMM format
        quantize_config = AWQConfig(
            bits=self.bits,
            group_size=self.group_size,
            sym=not self.zero_point,
        )

        self._model = GPTQModel.from_pretrained(
            self.model_name,
            quantize_config=quantize_config,
        )

        logger.info(f"Quantizing to {self.bits}-bit with {len(calib_data)} calibration blocks...")
        self._model.quantize(
            calib_data,
            tokenizer=self._tokenizer,
            batch_size=1,
            backend=BACKEND.AWQ_TORCH,  # avoid JIT-built CUDA kernels; checkpoint format is unchanged
        )

        return self._model

    def save(self, output_dir: str | Path | None = None) -> Path:
        """Save the quantized model."""
        if output_dir is None:
            output_dir = self._get_output_dir()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving quantized model to {output_dir}...")
        self._model.save_quantized(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))
        logger.info(f"Done! Model saved to {output_dir}")
        return output_dir

    def verify(self, model_path: str | Path) -> bool:
        """Verify the quantized model loads and generates correctly."""
        logger.info(f"Verifying quantized model at {model_path}...")
        try:
            model = GPTQModel.from_quantized(
                str(model_path),
                device_map="auto",
                backend=BACKEND.AWQ_TORCH,  # avoid JIT-built CUDA kernels (fail with system GCC 8.5)
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
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
