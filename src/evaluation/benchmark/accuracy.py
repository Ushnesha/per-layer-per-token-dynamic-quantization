"""Zero-shot HellaSwag accuracy using the standard LM Evaluation Harness task."""

import math
import os

from .perplexity import ENGINE_SETTINGS


def extract_accuracy(result: dict) -> dict:
    """Validate harness measurements and express scores and errors as percentages."""
    scores = result["results"]["hellaswag"]
    count = result["n-samples"]["hellaswag"]["effective"]
    if not isinstance(count, int) or isinstance(count, bool) or count <= 1:
        raise ValueError("HellaSwag must report at least two evaluated examples.")
    values = {}
    for key in ("acc,none", "acc_norm,none", "acc_stderr,none", "acc_norm_stderr,none"):
        value = float(scores[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid HellaSwag measurement: {key}={value}")
        values[key] = value * 100
    return {
        "status": "ok",
        "task": "hellaswag",
        "dataset": "Rowan/hellaswag",
        "split": "validation",
        "num_fewshot": 0,
        "apply_chat_template": False,
        "batch_size": 32,
        "seed": 0,
        "num_examples": count,
        "unit": "%",
        "accuracy": values["acc,none"],
        "normalized_accuracy": values["acc_norm,none"],
        "accuracy_stderr": values["acc_stderr,none"],
        "normalized_accuracy_stderr": values["acc_norm_stderr,none"],
        "task_version": result.get("versions", {}).get("hellaswag"),
        "scoring": "standard harness multiple-choice loglikelihood; raw and length-normalized",
    }


def evaluate_accuracy(model_path: str) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from lm_eval import simple_evaluate

    result = simple_evaluate(
        model="vllm",
        model_args={"pretrained": model_path, **ENGINE_SETTINGS},
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=32,
        limit=None,
        apply_chat_template=False,
        log_samples=False,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )
    if result is None:
        raise ValueError("LM Evaluation Harness returned no results.")
    return extract_accuracy(result)
