"""
Quantization base class and factory.

Provides a unified interface for all quantization methods.
Each quantizer implements: quantize(), save(), and verify().
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.config.settings import DEFAULT_MODEL_PATH, CALIBRATION_NUM_SAMPLES
from src.utils.logger import get_logger

logger = get_logger()


class BaseQuantizer(ABC):
    """Abstract base class for all quantization methods."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_PATH,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        sym: bool = True,
        calib_samples: int = CALIBRATION_NUM_SAMPLES,
    ):
        self.model_name = model_name
        self.bits = bits
        self.group_size = group_size
        self.zero_point = zero_point
        self.sym = sym
        self.calib_samples = calib_samples

        self._model = None
        self._tokenizer = None

    @abstractmethod
    def quantize(self, calib_data: List[str]) -> Any:
        """Run quantization on the model."""
        ...

    @abstractmethod
    def save(self, output_dir: str | Path) -> Path:
        """Save the quantized model."""
        ...

    @abstractmethod
    def verify(self, model_path: str | Path) -> bool:
        """Verify the quantized model loads and generates correctly."""
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the quantization method name (e.g., 'awq', 'gptq')."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}(bits={self.bits}, group_size={self.group_size})"


class QuantizerFactory:
    """Factory to create quantizers by method name."""

    _registry: Dict[str, type] = {}
    _lazy_registry: Dict[str, callable] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a quantizer class."""
        def wrapper(kls):
            cls._registry[name] = kls
            return kls
        return wrapper

    @classmethod
    def register_lazy(cls, name: str, loader: callable):
        """Register a lazy loader function for a quantizer.

        The loader is called only when the quantizer is first requested,
        avoiding ImportErrors from missing/incompatible libraries for
        unused quantization methods.
        """
        cls._lazy_registry[name] = loader

    @classmethod
    def create(cls, method: str, **kwargs) -> BaseQuantizer:
        """Create a quantizer instance by method name."""
        # Resolve lazy registration on first request
        if method in cls._lazy_registry and method not in cls._registry:
            try:
                cls._registry[method] = cls._lazy_registry[method]()
            except ImportError as e:
                raise ImportError(
                    f"Failed to import quantizer '{method}': {e}. "
                    f"Make sure the required library is installed."
                ) from e

        if method not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown quantization method: {method}. Available: {available}")
        return cls._registry[method](**kwargs)

    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered quantization methods."""
        return list(cls._registry.keys())
