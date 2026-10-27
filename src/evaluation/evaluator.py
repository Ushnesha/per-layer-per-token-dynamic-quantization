"""
Unified evaluation framework.

Runs the complete evaluation suite (perplexity, lm_eval, memory, latency)
on any quantized model and produces a single JSON results file.

Every method is measured the same way, on the same engine:
    - Perplexity + latency: vLLM, in an isolated subprocess (src.evaluation.vllm_bench).
    - HellaSwag: lm_eval CLI with the vLLM backend, in a second subprocess.
    - Memory: exact weight storage from the checkpoint's safetensors headers,
      plus the GPU weight memory vLLM reports at load time.
"""

import glob
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.settings import (
    QuantConfig, EvalConfig, DEFAULT_MODEL_PATH, PROJECT_ROOT, SEED,
    CALIBRATION_NUM_SAMPLES, CALIBRATION_SEQ_LEN, CALIBRATION_SOURCE, GROUP_SIZE,
    GPTQ_DAMP_PERCENT, SMOOTHQUANT_ALPHA,
    VLLM_DTYPE, VLLM_MAX_MODEL_LEN, VLLM_GPU_MEMORY_UTILIZATION,
)
from src.utils.logger import get_logger

logger = get_logger()

SUBPROCESS_TIMEOUT = 7200

# Same environment for every benchmark subprocess. FlashInfer's sampler is
# JIT-compiled with the host GCC (8.5 here, which fails); vLLM's PyTorch
# sampler is used instead for all methods.
BENCH_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def checkpoint_weight_gb(model_path: str) -> Optional[float]:
    """Exact bytes of all tensors in the checkpoint's top-level safetensors files (GB = 1e9 bytes)."""
    shards = sorted(Path(model_path).glob("*.safetensors"))
    if not shards:
        return None
    total = 0
    for shard in shards:
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name != "__metadata__":
                begin, end = info["data_offsets"]
                total += end - begin
    return round(total / 1e9, 3)


class Evaluator:
    """Run the full evaluation suite on a model."""

    def __init__(self, quant_config: QuantConfig, eval_config: Optional[EvalConfig] = None):
        self.quant = quant_config
        self.eval = eval_config or EvalConfig()
        self.results: Dict[str, Any] = {}
        self.log_dir = self.quant.results_dir / "logs"

    def _protocol(self) -> Dict[str, Any]:
        """Settings shared by every row of the comparison table (stored with the results)."""
        return {
            "seed": SEED,
            "calibration": {"source": CALIBRATION_SOURCE, "num_samples": CALIBRATION_NUM_SAMPLES,
                            "seq_len": CALIBRATION_SEQ_LEN},
            "method_config": {
                "fp16": {},
                "awq": {"group_size": GROUP_SIZE, "zero_point": True},
                "gptq": {"group_size": GROUP_SIZE, "sym": True, "desc_act": False,
                         "damp_percent": GPTQ_DAMP_PERCENT},
                "smoothquant": {"alpha": SMOOTHQUANT_ALPHA, "weights": "int8 per-channel",
                                "activations": "int8 dynamic per-token"},
            }[self.quant.method],
            "ppl": {"dataset": "wikitext-2-raw-v1 test", "seq_len": self.eval.ppl_seq_len,
                    "windows": "non-overlapping, BOS + seq_len-1 tokens"},
            "lm_eval": {"tasks": self.eval.tasks, "num_fewshot": self.eval.num_fewshot,
                        "batch_size": self.eval.batch_size, "backend": "vllm"},
            "latency": {"batch_size": 1, "input_len": self.eval.prompt_length,
                        "output_len": self.eval.output_length, "warmup": self.eval.warmup_iters,
                        "iters": self.eval.measure_iters, "statistic": "median"},
            "engine": {"name": "vllm", "dtype": VLLM_DTYPE, "max_model_len": VLLM_MAX_MODEL_LEN,
                       "gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
                       "prefix_caching": False},
        }

    def run(
        self,
        model_path: str | Path,
        run_ppl: bool = True,
        run_lm_eval: bool = True,
        run_memory: bool = True,
        run_latency: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full evaluation suite.

        Returns:
            Dictionary with all evaluation results.
        """
        model_path = str(model_path)
        logger.info(f"{'='*60}")
        logger.info(f"Evaluating: {self.quant.label} @ {model_path}")
        logger.info(f"{'='*60}")

        self.results = {
            "method": self.quant.method,
            "label": self.quant.label,
            "bits": self.quant.bits,
            "model_path": model_path,
            "model_name": DEFAULT_MODEL_PATH,
            "timestamp": datetime.now().isoformat(),
            "protocol": self._protocol(),
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if run_memory:
            self.results["memory"] = {"weight_storage_gb": checkpoint_weight_gb(model_path)}

        # PPL, latency and vLLM's measured weight memory share one engine instance
        if run_ppl or run_latency or run_memory:
            self._run_vllm_bench(model_path, run_ppl, run_latency, run_memory)

        if run_lm_eval:
            self._run_lm_eval(model_path)

        output_path = self._save_results()
        logger.info(f"Results saved to: {output_path}")
        self._print_summary()
        return self.results

    def _run_subprocess(self, cmd, log_name: str) -> subprocess.CompletedProcess:
        """Run a benchmark subprocess from the project root; keep full stdout/stderr in a log file."""
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT, cwd=str(PROJECT_ROOT),
            env={**os.environ, **BENCH_ENV},
        )
        log_path = self.log_dir / f"{self.quant.label.lower()}_{log_name}.log"
        with open(log_path, "w") as f:
            f.write(f"$ {' '.join(cmd)}\n\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
        logger.info(f"{log_name} log: {log_path}")
        return result

    # ── Perplexity + latency + vLLM weight memory (single subprocess) ─────
    def _run_vllm_bench(self, model_path: str, run_ppl: bool, run_latency: bool, run_memory: bool):
        logger.info("Running vLLM benchmark (perplexity / latency / memory)...")
        out_path = self.log_dir / f"{self.quant.label.lower()}_vllm_bench.json"
        cmd = [sys.executable, "-m", "src.evaluation.vllm_bench", "--model", model_path, "--out", str(out_path)]
        if not run_ppl:
            cmd.append("--skip-ppl")
        if not run_latency:
            cmd.append("--skip-latency")

        try:
            result = self._run_subprocess(cmd, "vllm_bench")
        except subprocess.TimeoutExpired:
            err = {"error": "timeout"}
            self._record_bench_error(err, run_ppl, run_latency)
            return

        if result.returncode != 0 or not out_path.exists():
            err = {"error": result.stderr[-1000:] if result.stderr else "unknown error"}
            logger.error(f"vLLM benchmark failed:\n{err['error']}")
            self._record_bench_error(err, run_ppl, run_latency)
            return

        with open(out_path) as f:
            bench = json.load(f)
        self.results["engine"] = bench["engine"]
        if run_ppl:
            self.results["perplexity"] = bench["perplexity"]
        if run_latency:
            self.results["latency"] = bench["latency"]
        if run_memory:
            match = re.search(r"Model loading took ([\d.]+) GiB", result.stderr + result.stdout)
            self.results["memory"]["vllm_weight_memory_gib"] = float(match.group(1)) if match else None

    def _record_bench_error(self, err: Dict[str, str], run_ppl: bool, run_latency: bool):
        if run_ppl:
            self.results["perplexity"] = err
        if run_latency:
            self.results["latency"] = err

    # ── lm_eval (vLLM backend, same for every method) ─────────────────────
    def _run_lm_eval(self, model_path: str):
        logger.info("Running lm_eval (vLLM backend)...")
        output_dir = self.quant.results_dir / "lm_eval" / self.quant.label.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        model_args = (
            f"pretrained={model_path},dtype={VLLM_DTYPE},max_model_len={VLLM_MAX_MODEL_LEN},"
            f"gpu_memory_utilization={VLLM_GPU_MEMORY_UTILIZATION},seed={SEED}"
        )
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "vllm",
            "--model_args", model_args,
            "--tasks", ",".join(self.eval.tasks),
            "--num_fewshot", str(self.eval.num_fewshot),
            "--batch_size", str(self.eval.batch_size),
            "--seed", str(SEED),
            "--output_path", str(output_dir),
        ]

        started = datetime.now().timestamp()
        try:
            result = self._run_subprocess(cmd, "lm_eval")
        except subprocess.TimeoutExpired:
            self.results["lm_eval"] = {"error": "timeout"}
            return

        if result.returncode != 0:
            self.results["lm_eval"] = {"error": result.stderr[-1000:]}
            logger.error(f"lm_eval failed:\n{result.stderr[-1000:]}")
            return

        json_files = [p for p in glob.glob(str(output_dir / "**" / "results_*.json"), recursive=True)
                      if Path(p).stat().st_mtime >= started]
        if not json_files:
            self.results["lm_eval"] = {"error": "no json output"}
            return
        with open(max(json_files, key=lambda p: Path(p).stat().st_mtime)) as f:
            self.results["lm_eval"] = json.load(f).get("results", {})
        logger.info("lm_eval results captured")

    # ── Save & Report ─────────────────────────────────────────────────────
    def _save_results(self) -> Path:
        """Save results to JSON file."""
        filename = f"{self.quant.label.lower()}.json"
        output_path = self.quant.results_dir / filename
        self.quant.results_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        return output_path

    def _print_summary(self):
        """Print a summary of results."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Results Summary: {self.quant.label}")
        logger.info(f"{'='*60}")

        ppl = self.results.get("perplexity", {})
        if "wikitext" in ppl:
            logger.info(f"  WikiText-2 PPL: {ppl['wikitext']}")

        hs = self.results.get("lm_eval", {}).get("hellaswag", {})
        if "acc_norm,none" in hs:
            logger.info(f"  HellaSwag acc_norm: {hs['acc_norm,none']:.4f} ± {hs.get('acc_norm_stderr,none', 0):.4f}")

        mem = self.results.get("memory", {})
        for k, v in mem.items():
            logger.info(f"  {k}: {v}")

        lat = self.results.get("latency", {})
        for k in ("ttft_ms", "tpot_ms", "e2e_ms"):
            if k in lat:
                logger.info(f"  {k}: {lat[k]}")

        for k, v in self.results.items():
            if isinstance(v, dict) and "error" in v:
                logger.info(f"  {k}: ERROR (see {self.log_dir})")

        logger.info(f"{'='*60}\n")
