"""
Per-Layer Per-Token LLM Serving — Quantization Benchmarking Framework

Modules:
    src.baselines       - Static quantization (AWQ, GPTQ, SmoothQuant)
    src.evaluation      - Evaluation suite (PPL, lm_eval, memory, latency)
    src.experiments     - Experiment runners and comparators
    src.utils           - Shared utilities (model loader, data, perplexity, logger)
    src.config          - Centralized configuration
"""

__version__ = "0.1.0"