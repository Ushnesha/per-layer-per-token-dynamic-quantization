"""
Static quantization baselines.

Quantizers:
    - AWQ (Activation-Aware Weight Quantization)
    - GPTQ (Hessian-aware Post-Training Quantization)
    - SmoothQuant (Activation smoothing + weight quantization)

Usage:
    from src.baselines.base_quantizer import QuantizerFactory

    quantizer = QuantizerFactory.create("awq", bits=4)
"""

from src.baselines.base_quantizer import BaseQuantizer, QuantizerFactory

# Lazy imports: quantizers are only loaded when requested by the factory.
# This avoids ImportErrors from missing/incompatible libraries for
# quantization methods the user isn't currently running.
def _import_awq():
    from src.baselines.awq_quantizer import AWQQuantizer
    return AWQQuantizer

def _import_gptq():
    from src.baselines.gptq_quantizer import GPTQQuantizer
    return GPTQQuantizer

def _import_smoothquant():
    from src.baselines.smoothquant_quantizer import SmoothQuantQuantizer
    return SmoothQuantQuantizer

# Register lazy loaders with the factory
QuantizerFactory.register_lazy("awq", _import_awq)
QuantizerFactory.register_lazy("gptq", _import_gptq)
QuantizerFactory.register_lazy("smoothquant", _import_smoothquant)

__all__ = [
    "BaseQuantizer",
    "QuantizerFactory",
]