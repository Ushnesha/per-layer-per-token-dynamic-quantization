#!/usr/bin/env python3
"""
Main entry point for the quantization benchmarking framework.

Usage:
    # Quantize with AWQ INT4
    python run.py quantize --method awq --bits 4

    # Evaluate FP16 baseline
    python run.py evaluate --method fp16

    # Evaluate AWQ INT4
    python run.py evaluate --method awq --bits 4

    # Compare all results
    python run.py compare

    # Full pipeline: quantize + evaluate
    python run.py pipeline --method awq --bits 4

    # Per-unit sensitivity profile for mixed precision
    python run.py profile

    # Mixed-precision GPTQ (base W4, listed units overridden) + evaluation
    python run.py quantize --method mixed --bits 4 --name lm8-down8 --assign lm_head=8 layers.1.down=8
    python run.py evaluate --method mixed --bits 4 --name lm8-down8

    # Same allocations on top of AWQ (compressed-tensors checkpoint; first run builds the AWQ base)
    python run.py quantize --method mixed-awq --bits 4 --name lm8 --assign lm_head=8
    python run.py quantize --method mixed-awq --bits 4 --budget 5.75
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Quantization Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── Quantize ────────────────────────────────────────────────────────
    quantize_parser = subparsers.add_parser("quantize", help="Run quantization")
    quantize_parser.add_argument("--method", type=str, nargs="+", default=["awq"],
                                 choices=["awq", "gptq", "smoothquant", "mixed", "mixed-awq"])
    quantize_parser.add_argument("--bits", type=int, nargs="+", default=[4], choices=[4, 8])
    quantize_parser.add_argument("--model", type=str, default=None)
    quantize_parser.add_argument("--group-size", type=int, default=128)
    quantize_parser.add_argument("--no-zero-point", action="store_true")
    quantize_parser.add_argument("--calib-samples", type=int, default=128)
    quantize_parser.add_argument("--calib-seq-len", type=int, default=2048)
    quantize_parser.add_argument("--calib-source", type=str, default="pile",
                                 choices=["pile", "wikitext"])
    quantize_parser.add_argument("--force", action="store_true")
    quantize_parser.add_argument("--verify", action="store_true")
    quantize_parser.add_argument("--name", type=str, default=None,
                                 help="Allocation name (mixed only)")
    quantize_parser.add_argument("--assign", type=str, nargs="+", default=None,
                                 help="Unit bit overrides for mixed, e.g. lm_head=8 layers.1.down=8")
    quantize_parser.add_argument("--budget", type=float, default=None,
                                 help="Weight storage budget in GB (mixed): allocation solved from the profile")
    quantize_parser.add_argument("--signal", type=str, default="reference",
                                 choices=["reference", "gptq", "cheap", "random", "position"],
                                 help="Allocation signal (mixed --budget); random/position are controls")
    quantize_parser.add_argument("--seed", type=int, default=0, help="Seed for --signal random")
    quantize_parser.add_argument("--metric", type=str, default="kl",
                                 help="kl | delta_nll (reference), abs | rel (cheap)")

    # ── Evaluate ────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation suite")
    eval_parser.add_argument("--method", type=str, nargs="+", default=["fp16"],
                             choices=["fp16", "awq", "gptq", "smoothquant", "mixed", "mixed-awq"])
    eval_parser.add_argument("--bits", type=int, nargs="+", default=[16])
    eval_parser.add_argument("--name", type=str, default=None, help="Allocation name (mixed only)")
    eval_parser.add_argument("--tasks", type=str, nargs="+", default=None,
                             help="lm_eval tasks (default: hellaswag); merged into existing results")
    eval_parser.add_argument("--model", type=str, default=None)
    eval_parser.add_argument("--skip-ppl", action="store_true")
    eval_parser.add_argument("--skip-lm-eval", action="store_true")
    eval_parser.add_argument("--skip-memory", action="store_true")
    eval_parser.add_argument("--skip-latency", action="store_true")

    # ── Compare ─────────────────────────────────────────────────────────
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("--results-dir", type=str, default="results")
    compare_parser.add_argument("--labels", type=str, nargs="+", default=None)
    compare_parser.add_argument("--export-md", type=str, default=None)
    compare_parser.add_argument("--export-json", type=str, default=None)

    # ── Profile (per-unit sensitivity for mixed precision) ──────────────
    profile_parser = subparsers.add_parser("profile", help="Per-unit sensitivity profiling")
    profile_parser.add_argument("--model", type=str, default=None)
    profile_parser.add_argument("--bits", type=int, nargs="+", default=[4, 8], choices=[4, 8])
    profile_parser.add_argument("--calib-samples", type=int, default=32)
    profile_parser.add_argument("--ref-samples", type=int, default=8)
    profile_parser.add_argument("--calib-seq-len", type=int, default=2048)
    profile_parser.add_argument("--calib-source", type=str, default="pile",
                                choices=["pile", "wikitext"])
    profile_parser.add_argument("--skip-reference", action="store_true")
    profile_parser.add_argument("--no-lm-head", action="store_true")
    profile_parser.add_argument("--gptq-reference", action="store_true",
                                help="Add KL with GPTQ checkpoint weights to the existing profile")
    profile_parser.add_argument("--out", type=str, default=None)

    # ── Pipeline (quantize + evaluate) ──────────────────────────────────
    pipeline_parser = subparsers.add_parser("pipeline", help="Quantize then evaluate")
    pipeline_parser.add_argument("--method", type=str, nargs="+", default=["awq"],
                                 choices=["awq", "gptq", "smoothquant"])
    pipeline_parser.add_argument("--bits", type=int, nargs="+", default=[4], choices=[4, 8])
    pipeline_parser.add_argument("--model", type=str, default=None)
    pipeline_parser.add_argument("--force", action="store_true")
    pipeline_parser.add_argument("--verify", action="store_true")
    pipeline_parser.add_argument("--skip-ppl", action="store_true")
    pipeline_parser.add_argument("--skip-lm-eval", action="store_true")
    pipeline_parser.add_argument("--skip-memory", action="store_true")
    pipeline_parser.add_argument("--skip-latency", action="store_true")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "quantize":
        from src.experiments.run_quantization import run
        run(args)

    elif args.command == "evaluate":
        from src.experiments.run_evaluation import run
        run(args)

    elif args.command == "compare":
        from src.experiments.compare_results import run
        run(args)

    elif args.command == "profile":
        from src.adaptive.sensitivity import run
        run(args)

    elif args.command == "pipeline":
        # Quantize first
        from src.experiments.run_quantization import run as run_quantize
        quantize_args = argparse.Namespace(
            method=args.method,
            bits=args.bits,
            model=args.model,
            group_size=128,
            no_zero_point=False,
            calib_samples=128,
            calib_seq_len=2048,
            calib_source="pile",
            force=args.force,
            verify=args.verify,
        )
        run_quantize(quantize_args)

        # Then evaluate
        from src.experiments.run_evaluation import run as run_evaluation
        eval_args = argparse.Namespace(
            method=args.method,
            bits=args.bits,
            model=args.model,
            skip_ppl=args.skip_ppl,
            skip_lm_eval=args.skip_lm_eval,
            skip_memory=args.skip_memory,
            skip_latency=args.skip_latency,
        )
        run_evaluation(eval_args)


if __name__ == "__main__":
    main()
