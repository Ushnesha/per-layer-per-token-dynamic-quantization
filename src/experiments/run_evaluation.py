"""
Run the complete evaluation suite on any quantized model.

Usage:
    # Evaluate FP16 baseline
    python -m src.experiments.run_evaluation --method fp16

    # Evaluate AWQ-INT4
    python -m src.experiments.run_evaluation --method awq --bits 4

    # Evaluate all methods
    python -m src.experiments.run_evaluation --method fp16 awq gptq smoothquant --bits 16 4 4 8

    # Skip lm_eval (faster)
    python -m src.experiments.run_evaluation --method awq --bits 4 --skip-lm-eval
"""

import argparse
import sys
from pathlib import Path

from src.config.settings import QuantConfig, EvalConfig, DEFAULT_MODEL_PATH
from src.evaluation.evaluator import Evaluator
from src.utils.logger import get_logger

logger = get_logger()


def run(args):
    """Run evaluation on specified methods."""
    bits_list = args.bits if args.bits else [16]
    methods = args.method

    # If single bits value but multiple methods, broadcast
    if len(bits_list) == 1 and len(methods) > 1:
        bits_list = bits_list * len(methods)

    for i, method in enumerate(methods):
        bits = bits_list[i] if i < len(bits_list) else 16

        quant_config = QuantConfig(method=method, bits=bits)
        eval_config = EvalConfig()

        # Determine model path
        if method == "fp16":
            model_path = args.model or DEFAULT_MODEL_PATH
        else:
            model_path = str(quant_config.output_dir)

        # Check model exists
        if method != "fp16" and not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            logger.error(f"Run quantization first: python -m src.experiments.run_quantization --method {method} --bits {bits}")
            continue

        evaluator = Evaluator(quant_config, eval_config)
        results = evaluator.run(
            model_path=model_path,
            run_ppl=not args.skip_ppl,
            run_lm_eval=not args.skip_lm_eval,
            run_memory=not args.skip_memory,
            run_latency=not args.skip_latency,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation suite on quantized models")
    parser.add_argument("--method", type=str, nargs="+", default=["fp16"],
                        choices=["fp16", "awq", "gptq", "smoothquant"],
                        help="Quantization method(s) to evaluate")
    parser.add_argument("--bits", type=int, nargs="+", default=[16],
                        help="Bit-width(s) for quantization")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (default: Llama-3-8B-Instruct)")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-lm-eval", action="store_true", help="Skip lm_eval harness")
    parser.add_argument("--skip-memory", action="store_true", help="Skip memory benchmark")
    parser.add_argument("--skip-latency", action="store_true", help="Skip latency benchmark")
    args = parser.parse_args()
    run(args)
