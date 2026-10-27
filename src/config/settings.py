"""
Centralized configuration for the quantization benchmarking framework.

All paths, model names, evaluation tasks, and quantization settings
are defined here to avoid duplication across modules.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── Project roots ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# ── Model configuration ────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_MODEL_PATH = MODELS_DIR / "llama-3-8b-instruct"

# Prefer local model if it exists, fall back to HF hub ID
if LOCAL_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = str(LOCAL_MODEL_PATH)
else:
    DEFAULT_MODEL_PATH = DEFAULT_MODEL_NAME

# ── Dataset configuration ──────────────────────────────────────────────────
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
HELLASWAG_DATASET = "Rowan/hellaswag"
PILEVAL_DATASET = "mit-han-lab/pile-val-backup"

# ── Shared benchmark protocol ──────────────────────────────────────────────
# Every method is calibrated and evaluated with these same settings so that
# rows in the comparison table differ only in the quantization method.
SEED = 0

# Calibration: same token blocks for AWQ, GPTQ and SmoothQuant.
# Pile-val (as in the AWQ paper) rather than WikiText, so WikiText PPL stays held-out.
CALIBRATION_NUM_SAMPLES = 128
CALIBRATION_SEQ_LEN = 2048
CALIBRATION_SOURCE = "pile"

# Method configs (each method's standard deployable config; vLLM serves all
# of these with prebuilt Marlin / INT8 CUTLASS kernels).
#   AWQ  W4A16: g128, asymmetric (zero-point)        -> awq_marlin
#   GPTQ W4A16: g128, symmetric, no act-order        -> gptq_marlin (sym only)
#   SmoothQuant W8A8: per-channel W, per-token dynamic A, alpha=0.8 -> int8 cutlass
# lm_head and embeddings stay FP16 for all methods.
GROUP_SIZE = 128
GPTQ_DAMP_PERCENT = 0.01
SMOOTHQUANT_ALPHA = 0.8

# Perplexity: WikiText-2 test, non-overlapping windows over the full split.
PPL_SEQ_LEN = 2048

# ── Evaluation configuration ───────────────────────────────────────────────
EVAL_TASKS = ["hellaswag"]
LM_EVAL_NUM_FEWSHOT = 0
LM_EVAL_BATCH_SIZE = "32"
DEFAULT_DEVICE = "cuda:0"

# vLLM engine settings, identical for every method.
VLLM_DTYPE = "float16"
VLLM_MAX_MODEL_LEN = 4096
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# ── Quantization presets ───────────────────────────────────────────────────
AWQ_OUTPUT_BASE = MODELS_DIR / "awq"
GPTQ_OUTPUT_BASE = MODELS_DIR / "gptq"
SMOOTHQUANT_OUTPUT_BASE = MODELS_DIR / "smoothquant"

# ── Dataclasses ────────────────────────────────────────────────────────────
@dataclass
class QuantConfig:
    """Configuration for a single quantization run."""
    method: str  # "fp16" | "awq" | "gptq" | "smoothquant"
    bits: int = 16  # 16 for FP16, 8 for INT8, 4 for INT4
    group_size: int = GROUP_SIZE
    zero_point: bool = True
    version: str = "gemm"  # AWQ format: "gemm" | "gemv"
    sym: bool = True  # Symmetric quantization (GPTQ)

    @property
    def label(self) -> str:
        if self.method == "fp16":
            return "fp16"
        suffix = "a16" if self.method in ("awq", "gptq") else "a8"
        return f"{self.method.upper()}-w{self.bits}{suffix}"

    @property
    def output_dir(self) -> Path:
        base = {
            "awq": AWQ_OUTPUT_BASE,
            "gptq": GPTQ_OUTPUT_BASE,
            "smoothquant": SMOOTHQUANT_OUTPUT_BASE,
        }.get(self.method, MODELS_DIR)
        return base / f"llama3-8b-instruct-w{self.bits}a{'16' if self.method != 'smoothquant' else '8'}"

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.method


@dataclass
class EvalConfig:
    """Configuration for evaluation runs."""
    tasks: List[str] = field(default_factory=lambda: EVAL_TASKS)
    num_fewshot: int = LM_EVAL_NUM_FEWSHOT
    batch_size: str = LM_EVAL_BATCH_SIZE
    device: str = DEFAULT_DEVICE
    ppl_seq_len: int = PPL_SEQ_LEN
    calib_samples: int = CALIBRATION_NUM_SAMPLES

    # Latency (batch 1, single request, prefix caching off)
    prompt_length: int = 512
    output_length: int = 128
    warmup_iters: int = 3
    measure_iters: int = 10
