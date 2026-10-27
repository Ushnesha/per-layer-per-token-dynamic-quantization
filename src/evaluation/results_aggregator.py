"""
Results aggregator and comparator.

Loads result JSON files from different quantization methods and produces
comparison tables and summaries.

Usage:
    from src.evaluation.results_aggregator import ResultsAggregator

    agg = ResultsAggregator("results/")
    agg.compare(["fp16", "awq-int4", "gptq-int4", "smoothquant-w8a8"])
    agg.print_table()
    agg.export_markdown("results/comparison.md")
"""

import json
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger()


class ResultsAggregator:
    """Load, compare, and export results from evaluation runs."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results: Dict[str, Dict[str, Any]] = {}

    def load(self, label: str, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Load a single results file."""
        if filepath:
            path = Path(filepath)
        else:
            # Find the latest file for this label (evaluator writes <label>.json)
            files = []
            for name in (f"{label.lower()}.json", f"{label.lower()}_*.json"):
                files += glob.glob(str(self.results_dir / "**" / name), recursive=True)
            files = [f for f in files if "lm_eval" not in Path(f).parts and "logs" not in Path(f).parts]
            if not files:
                logger.warning(f"No results found for label: {label}")
                self.results[label] = {}
                return self.results[label]
            path = Path(max(files, key=lambda x: Path(x).stat().st_mtime))

        with open(path) as f:
            data = json.load(f)
        self.results[label] = data
        logger.info(f"Loaded results: {label} from {path}")
        return data

    def load_all(self, labels: List[str]):
        """Load results for multiple labels."""
        for label in labels:
            self.load(label)

    def get_metric(self, label: str, *keys: str) -> Any:
        """Get a nested metric value by key path."""
        data = self.results.get(label, {})
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, {})
            else:
                return None
        return data

    def get_ppl(self, label: str) -> Optional[float]:
        """Get WikiText perplexity for a label."""
        ppl = self.get_metric(label, "perplexity", "wikitext")
        return round(ppl, 4) if ppl else None

    def get_hellaswag(self, label: str) -> Optional[float]:
        """Get HellaSwag length-normalized accuracy (acc_norm) for a label."""
        val = self.get_metric(label, "lm_eval", "hellaswag", "acc_norm,none")
        return round(val, 4) if isinstance(val, float) else None

    def get_weight_memory(self, label: str) -> Optional[float]:
        """Get weight storage in GB (exact checkpoint tensor bytes)."""
        mem = self.get_metric(label, "memory", "weight_storage_gb")
        return mem if isinstance(mem, (int, float)) else None

    def get_tpots(self, label: str) -> Dict[str, float]:
        """Get median TPOT (batch 1, vLLM) keyed by input/output length."""
        lat = self.get_metric(label, "latency") or {}
        if "tpot_ms" not in lat:
            return {}
        return {f"in{lat['input_len']}_out{lat['output_len']}": lat["tpot_ms"]}

    def compare(self, labels: List[str]) -> Dict[str, Any]:
        """
        Compare multiple quantization methods.

        Returns a dictionary with comparison data.
        """
        self.load_all(labels)

        comparison = {
            "labels": labels,
            "perplexity": {},
            "hellaswag": {},
            "weight_memory_gb": {},
            "tpot_ms": {},
            "timestamp": datetime.now().isoformat(),
        }

        for label in labels:
            comparison["perplexity"][label] = self.get_ppl(label)
            comparison["hellaswag"][label] = self.get_hellaswag(label)
            comparison["weight_memory_gb"][label] = self.get_weight_memory(label)
            comparison["tpot_ms"][label] = self.get_tpots(label)

        return comparison

    def print_table(self, comparison: Optional[Dict[str, Any]] = None):
        """Print a formatted comparison table."""
        if comparison is None:
            comparison = self.compare(list(self.results.keys()))

        labels = comparison["labels"]

        # Header
        col_width = 18
        header = f"{'Metric':<{col_width}}"
        for label in labels:
            header += f"{label:^{col_width}}"
        print(f"\n{'='*col_width*(len(labels)+1)}")
        print(header)
        print(f"{'-'*col_width*(len(labels)+1)}")

        # Perplexity
        row = f"{'WikiText PPL':<{col_width}}"
        for label in labels:
            val = comparison["perplexity"].get(label)
            row += f"{val if val else 'N/A':^{col_width}}"
        print(row)

        # HellaSwag
        row = f"{'HellaSwag AccNorm':<{col_width}}"
        for label in labels:
            val = comparison["hellaswag"].get(label)
            row += f"{val if val else 'N/A':^{col_width}}"
        print(row)

        # Weight Memory
        row = f"{'Weight Mem (GB)':<{col_width}}"
        for label in labels:
            val = comparison["weight_memory_gb"].get(label)
            row += f"{val if val else 'N/A':^{col_width}}"
        print(row)

        # TPOT (first available)
        row = f"{'TPOT (ms/token)':<{col_width}}"
        for label in labels:
            tpots = comparison["tpot_ms"].get(label, {})
            if tpots:
                first = list(tpots.values())[0]
                row += f"{first:^{col_width}}"
            else:
                row += f"{'N/A':^{col_width}}"
        print(row)

        print(f"{'='*col_width*(len(labels)+1)}\n")

    def export_markdown(self, filepath: str, comparison: Optional[Dict[str, Any]] = None):
        """Export comparison table as Markdown."""
        if comparison is None:
            comparison = self.compare(list(self.results.keys()))

        labels = comparison["labels"]

        lines = []
        lines.append(f"# Quantization Comparison Report")
        lines.append(f"\n**Generated**: {comparison['timestamp']}")
        lines.append("")

        # Perplexity
        lines.append("## WikiText-2 Perplexity")
        lines.append("")
        lines.append("| Method | PPL |")
        lines.append("|--------|-----|")
        for label in labels:
            val = comparison["perplexity"].get(label, "N/A")
            lines.append(f"| {label} | {val} |")
        lines.append("")

        # HellaSwag
        lines.append("## HellaSwag Accuracy (acc_norm, 0-shot)")
        lines.append("")
        lines.append("| Method | Accuracy |")
        lines.append("|--------|----------|")
        for label in labels:
            val = comparison["hellaswag"].get(label, "N/A")
            lines.append(f"| {label} | {val} |")
        lines.append("")

        # Memory
        lines.append("## Weight Memory (GB)")
        lines.append("")
        lines.append("| Method | Memory (GB) |")
        lines.append("|--------|-------------|")
        for label in labels:
            val = comparison["weight_memory_gb"].get(label, "N/A")
            lines.append(f"| {label} | {val} |")
        lines.append("")

        # Latency
        lines.append("## TPOT (ms/token)")
        lines.append("")
        lines.append("| Method | TPOT (ms) |")
        lines.append("|--------|-----------|")
        for label in labels:
            tpots = comparison["tpot_ms"].get(label, {})
            if tpots:
                vals = ", ".join(f"{v}" for v in tpots.values())
                lines.append(f"| {label} | {vals} |")
            else:
                lines.append(f"| {label} | N/A |")
        lines.append("")

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to: {output_path}")
        return output_path

    def export_json(self, filepath: str, comparison: Optional[Dict[str, Any]] = None):
        """Export comparison data as JSON."""
        if comparison is None:
            comparison = self.compare(list(self.results.keys()))

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        logger.info(f"JSON comparison saved to: {output_path}")
        return output_path
