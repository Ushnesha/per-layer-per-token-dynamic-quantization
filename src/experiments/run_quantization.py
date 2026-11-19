"""
Run quantization on the base model.

Usage:
    # AWQ INT4
    python -m src.experiments.run_quantization --method awq --bits 4

    # AWQ INT8
    python -m src.experiments.run_quantization --method awq --bits 8

    # GPTQ INT4
    python -m src.experiments.run_quantization --method gptq --bits 4

    # SmoothQuant W8A8
    python -m src.experiments.run_quantization --method smoothquant

    # Force re-quantization
    python -m src.experiments.run_quantization --method awq --bits 4 --force
"""

import argparse
import shutil
import sys
from pathlib import Path

from transformers import AutoTokenizer

from src.config.settings import (
    DEFAULT_MODEL_PATH, CALIBRATION_NUM_SAMPLES, CALIBRATION_SEQ_LEN, CALIBRATION_SOURCE, GROUP_SIZE, SEED,
    MIXED_METHODS,
)
from src.baselines.base_quantizer import QuantizerFactory
from src.utils.data_loader import get_calibration_data
from src.utils.logger import get_logger

logger = get_logger()


def run(args):
    """Run quantization pipeline."""
    for method in args.method:
        for bits in args.bits:
            if method == "smoothquant" and bits != 8:
                logger.info(f"SmoothQuant only supports INT8, overriding bits={bits} -> 8")
                bits = 8

            act_bits = 8 if method == "smoothquant" else 16
            model_name = args.model or DEFAULT_MODEL_PATH
            logger.info(f"{'='*60}")
            logger.info(f"Quantizing: {method.upper()} W{bits}A{act_bits}")
            logger.info(f"{'='*60}")

            # Mixed precision: base bits plus per-unit overrides
            extra = {}
            if method in MIXED_METHODS and getattr(args, "budget", None):
                from src.adaptive.allocator import Allocator, overrides
                if args.signal in ("random", "position"):
                    # Sensitivity-free control at the same budget
                    allocator = Allocator()
                    allocation = allocator.solve_by_order(args.budget, allocator.control_order(args.signal, args.seed))
                    tag = f"rand{args.seed}" if args.signal == "random" else "pos"
                    meta = {"signal": args.signal, "seed": args.seed}
                else:
                    # Sensitivity-guided: solve the allocation for a weight-storage budget
                    allocator = Allocator(signal=args.signal, metric=args.metric)
                    allocation = allocator.solve(args.budget)
                    tag = f"{args.signal[:3]}-{args.metric}"
                    meta = {"signal": args.signal, "metric": args.metric}
                args.name = args.name or f"{tag}-{args.budget:.2f}"
                # Predicted reference KL for every allocation, so all variants are comparable
                reference = allocator if args.signal == "reference" else Allocator()
                extra = {"name": args.name, "assignments": overrides(allocation, base_bits=bits),
                         "meta": {"budget_gb": args.budget, **meta,
                                  "predicted_gb": round(allocator.size_gb(allocation), 4),
                                  "predicted_cost": allocator.predicted_cost(allocation),
                                  "predicted_reference_kl": reference.predicted_cost(allocation)}}
            elif method in MIXED_METHODS:
                from src.adaptive.allocation import parse_assignments
                if not getattr(args, "name", None):
                    raise ValueError(f"--name (with --assign) or --budget is required for --method {method}")
                extra = {"name": args.name, "assignments": parse_assignments(args.assign)}

            # Create quantizer
            quantizer = QuantizerFactory.create(
                method=method,
                model_name=model_name,
                bits=bits,
                group_size=args.group_size,
                zero_point=not args.no_zero_point,
                calib_samples=args.calib_samples,
                **extra,
            )

            # Check if already quantized
            output_dir = quantizer._get_output_dir() if hasattr(quantizer, '_get_output_dir') else None
            if output_dir and output_dir.exists() and any(output_dir.iterdir()) and not args.force:
                logger.info(f"Quantized model already exists at {output_dir}")
                logger.info("Use --force to re-quantize.")
                if len(args.method) == 1 and len(args.bits) == 1:
                    return
                continue

            # Force: remove existing
            if args.force and output_dir and output_dir.exists():
                logger.info(f"Removing existing: {output_dir}")
                shutil.rmtree(output_dir)

            # Load calibration data (identical token blocks for every method)
            logger.info("Loading calibration data...")
            calib_data = get_calibration_data(
                AutoTokenizer.from_pretrained(model_name),
                num_samples=args.calib_samples,
                seq_len=args.calib_seq_len,
                source=args.calib_source,
                seed=SEED,
            )

            # Quantize
            logger.info("Starting quantization...")
            quantizer.quantize(calib_data)

            # Save
            saved_path = quantizer.save()
            logger.info(f"Model saved to: {saved_path}")

            # Verify
            if args.verify:
                success = quantizer.verify(saved_path)
                if not success:
                    logger.error(f"Verification failed for {method} W{bits}A{act_bits}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quantization on Llama-3-8B-Instruct")
    parser.add_argument("--method", type=str, nargs="+", default=["awq"],
                        choices=["awq", "gptq", "smoothquant"],
                        help="Quantization method(s)")
    parser.add_argument("--bits", type=int, nargs="+", default=[4],
                        choices=[4, 8],
                        help="Bit-width(s) for quantization")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (default: Llama-3-8B-Instruct)")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE,
                        help="Group size for quantization")
    parser.add_argument("--no-zero-point", action="store_true",
                        help="Disable zero-point calibration")
    parser.add_argument("--calib-samples", type=int, default=CALIBRATION_NUM_SAMPLES,
                        help="Number of calibration blocks")
    parser.add_argument("--calib-seq-len", type=int, default=CALIBRATION_SEQ_LEN,
                        help="Tokens per calibration block")
    parser.add_argument("--calib-source", type=str, default=CALIBRATION_SOURCE,
                        choices=["pile", "wikitext"],
                        help="Calibration data source")
    parser.add_argument("--force", action="store_true",
                        help="Force re-quantization")
    parser.add_argument("--verify", action="store_true",
                        help="Run verification after quantization")
    args = parser.parse_args()
    run(args)
