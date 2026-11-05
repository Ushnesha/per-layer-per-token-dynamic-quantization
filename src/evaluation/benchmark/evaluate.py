#!/usr/bin/env python3
"""Independent evaluation CLI, built incrementally on dev/aditya.

Run with: python -m src.evaluation.benchmark.evaluate --model PATH

Writes evaluation measurements to JSON and a companion Markdown report.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence


def local_model_path(value: str) -> Path:
    """Accept an existing checkpoint directory containing config.json."""
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"model must be an existing local checkpoint directory: {path}"
        )
    if not (path / "config.json").is_file():
        raise argparse.ArgumentTypeError(
            f"model directory must contain config.json: {path}"
        )
    return path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local model checkpoint and write JSON and Markdown reports."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=local_model_path,
        metavar="PATH",
        help="local model checkpoint directory containing config.json",
    )
    parser.add_argument('--reference', type=lambda value: Path(value).expanduser().resolve(), metavar='JSON',
                        help='saved compatible FP16 report for quality/compression comparisons; default: automatic selection')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    from .protocol import PROFILE, SCOPE
    from .perplexity import ENGINE_SETTINGS
    from .worker import run_metric
    from .quality_tradeoff import describe_checkpoint, evaluate_quality_tradeoff, validate_reference
    from .compression_ratio import evaluate_compression_ratio
    from .adaptive_overhead import evaluate_adaptive_overhead
    from .tpot import evaluate_tpot
    from .e2e_latency import evaluate_e2e_latency
    from .throughput import evaluate_throughput
    from .goodput import evaluate_goodput
    from .max_concurrency import evaluate_max_concurrency
    from .report import write_markdown_report

    timestamp = datetime.now(timezone.utc)
    versions = {}
    for package in ("vllm", "transformers", "datasets", "torch", "torchcodec", "torchvision",
                    "torchaudio", "huggingface_hub", "lm-eval", "rouge", "nvidia-ml-py", "psutil"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    results = {
        "model": str(args.model),
        "timestamp": timestamp.isoformat(),
        "backend": "vllm",
        "evaluation_profile": PROFILE,
        "evaluation_scope": SCOPE,
        "runtime_settings": {"worker_multiprocessing": os.environ['VLLM_WORKER_MULTIPROC_METHOD']},
        "engine_settings": ENGINE_SETTINGS,
        "versions": versions,
        "metrics": {},
        "checkpoint": describe_checkpoint(args.model),
    }
    if args.reference:
        validate_reference(results, args.reference)
    results['requested_reference'] = str(args.reference) if args.reference else None
    output_dir = Path(__file__).resolve().parents[3] / 'results' / 'benchmark'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"evaluation_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    results['run_status'] = 'running'
    def save_progress():
        temporary = output_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
        temporary.replace(output_path)
        write_markdown_report(results, output_path)
    save_progress()
    failed = False
    # Validate worker memory instrumentation before the expensive dataset runs.
    stages = ('memory', 'max_context', 'perplexity', 'accuracy', 'retrieval',
              'long_context', 'kernel_time', 'ttft')
    for name in stages:
        results['active_stage'] = name
        save_progress()
        print(f"Running {name}...", flush=True)
        try:
            metric = run_metric(name, str(args.model))
        except Exception as error:
            metric = {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        if name == 'memory':
            for part in ('peak_memory', 'component_memory'):
                results['metrics'][part] = metric.get(part, {'status': 'failed', 'error': metric.get('error', 'Combined memory measurement unavailable.')})
        else:
            results["metrics"][name] = metric
        if metric["status"] == "failed":
            failed = True
            print(f"{name} failed: {metric.get('error', 'unknown error')}", file=sys.stderr)
        save_progress()

    try:
        results['metrics']['max_concurrency'] = evaluate_max_concurrency(results['metrics'].get('ttft', {}))
    except Exception as error:
        failed = True
        results['metrics']['max_concurrency'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['goodput'] = evaluate_goodput(results['metrics'].get('ttft', {}))
    except Exception as error:
        failed = True
        results['metrics']['goodput'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['throughput'] = evaluate_throughput(results['metrics'].get('ttft', {}))
    except Exception as error:
        failed = True
        results['metrics']['throughput'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['e2e_latency'] = evaluate_e2e_latency(results['metrics'].get('ttft', {}))
    except Exception as error:
        failed = True
        results['metrics']['e2e_latency'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['tpot'] = evaluate_tpot(results['metrics'].get('ttft', {}))
    except Exception as error:
        failed = True
        results['metrics']['tpot'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['adaptive_overhead'] = evaluate_adaptive_overhead(results, output_dir)
    except Exception as error:
        failed = True
        results['metrics']['adaptive_overhead'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['quality_tradeoff'] = evaluate_quality_tradeoff(results, output_dir, args.reference)
    except Exception as error:
        failed = True
        results['metrics']['quality_tradeoff'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    try:
        results['metrics']['compression_ratio'] = evaluate_compression_ratio(results, output_dir, args.reference)
    except Exception as error:
        failed = True
        results['metrics']['compression_ratio'] = {
            'status': 'failed', 'error_type': type(error).__name__, 'error': str(error),
        }
    core = ('perplexity', 'accuracy', 'retrieval', 'long_context', 'peak_memory', 'component_memory',
            'ttft', 'max_context', 'tpot', 'e2e_latency', 'throughput', 'goodput', 'max_concurrency')
    checks = {name: results['metrics'].get(name, {}).get('status') == 'ok' for name in core}
    checks['runtime_components'] = results['metrics'].get('component_memory', {}).get('runtime', {}).get('status') == 'ok'
    runtime = results['metrics'].get('component_memory', {}).get('runtime', {})
    checks['live_kv_occupancy'] = runtime.get('live_kv_occupancy', {}).get('status') == 'ok'
    checks['temporary_allocation_tracking'] = runtime.get('temporary_workspaces', {}).get('status') == 'ok'
    serving = results['metrics'].get('ttft', {})
    workload = serving.get('workload', {})
    closed = serving.get('trials', [])
    rates = serving.get('request_rate_trials', [])
    repetitions = workload.get('repetitions', 0)
    expected_closed = len(workload.get('prompt_lengths', [])) * len(workload.get('concurrency_levels', [])) * repetitions
    expected_rates = len(workload.get('prompt_lengths', [])) * len(workload.get('request_rates', [])) * repetitions
    checks['serving_sweep_complete'] = (expected_closed > 0 and expected_rates > 0
        and len(closed) == expected_closed and len(rates) == expected_rates
        and all(row.get('status') == 'ok' and row.get('warmup_failed') == 0
                and len(row.get('requests', [])) == workload.get('measured_requests_per_trial')
                and all(request.get('status') == 'ok' for request in row['requests']) for row in [*closed, *rates]))
    checks['context_verified'] = results['metrics'].get('max_context', {}).get('max_total_tokens') is not None
    results['evaluation_checks'] = {'status': 'ok' if all(checks.values()) else 'failed', 'checks': checks}
    failed = failed or not all(checks.values())
    incomplete = any(metric.get('status') not in ('ok', 'skipped', 'not_applicable') for metric in results['metrics'].values())
    incomplete = incomplete or any(item.get('status') != 'ok'
        for item in results['metrics'].get('compression_ratio', {}).get('components', {}).values())
    incomplete = incomplete or results['metrics'].get('kernel_time', {}).get('individual_kernels', {}).get('status') != 'ok'
    results['run_status'] = 'failed' if failed else 'partial' if incomplete else 'ok'
    results['metric_coverage'] = {name: metric.get('status') for name, metric in results['metrics'].items()}
    results['active_stage'] = None
    results['finished_at'] = datetime.now(timezone.utc).isoformat()
    temporary = output_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    temporary.replace(output_path)
    print(f"Results saved to: {output_path}")
    try:
        report_path = write_markdown_report(results, output_path)
        print(f"Readable report saved to: {report_path}")
    except (OSError, TypeError, ValueError) as error:
        failed = True
        print(f"Markdown report could not be written: {error}. JSON results are saved.", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
