"""
Compare results from all quantization methods and export reports.

Usage:
    # Compare all methods
    python -m src.experiments.compare_results

    # Compare specific methods
    python -m src.experiments.compare_results --labels fp16 awq-int4 gptq-int4

    # Export to markdown
    python -m src.experiments.compare_results --export-md results/comparison.md

    # Export to JSON
    python -m src.experiments.compare_results --export-json results/comparison.json
"""

import argparse
from pathlib import Path

from src.evaluation.results_aggregator import ResultsAggregator
from src.utils.logger import get_logger

logger = get_logger()


def run(args):
    """Compare and export results."""
    aggregator = ResultsAggregator(args.results_dir)

    # Determine labels to compare
    if args.labels:
        labels = args.labels
    else:
        # Auto-discover all result files
        labels = []
        results_path = Path(args.results_dir)
        for method_dir in ["fp16", "awq", "gptq", "smoothquant"]:
            method_path = results_path / method_dir
            if method_path.exists():
                import glob
                files = glob.glob(str(method_path / "*.json"))
                for f in files:
                    fname = Path(f).stem
                    # Extract label from filename (e.g., "awq-int4_2024..." -> "awq-int4")
                    parts = fname.rsplit("_", 1)
                    if parts:
                        label = parts[0]
                        if label not in labels:
                            labels.append(label)

        if not labels:
            logger.warning("No results found. Run evaluations first.")
            return

    logger.info(f"Comparing: {labels}")
    comparison = aggregator.compare(labels)

    # Print table
    aggregator.print_table(comparison)

    # Export if requested
    if args.export_md:
        aggregator.export_markdown(args.export_md, comparison)
    if args.export_json:
        aggregator.export_json(args.export_json, comparison)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare quantization results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing result files")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels to compare (auto-discovered if not specified)")
    parser.add_argument("--export-md", type=str, default=None,
                        help="Export comparison as Markdown to this path")
    parser.add_argument("--export-json", type=str, default=None,
                        help="Export comparison as JSON to this path")
    args = parser.parse_args()
    run(args)
