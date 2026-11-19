"""Selected LongBench v1 tasks with official task prompts and scoring rules.

Context-only token truncation is an explicit protocol variant, not an exact
reproduction of the original model-specific LongBench inference scripts.
Prompt/scorer sources: https://github.com/THUDM/LongBench/tree/main/LongBench
Copyright (c) 2023 THU-KEG & Zhipu AI; see LONGBENCH_LICENSE.txt.
"""

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import string
import zipfile

from .perplexity import ENGINE_SETTINGS
from .instruction_prompt import chat_parts, prompt_protocol, stop_token_ids

DATASET = "THUDM/LongBench"
CONTEXT_LENGTHS = (2048, 4096, 7168)
TASKS = {
    "qasper": {"metric": "token_f1", "max_tokens": 128, "prompt": (
        'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable".\n'
        'Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:'
    )},
    "hotpotqa": {"metric": "token_f1", "max_tokens": 32, "prompt": (
        "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    )},
    "gov_report": {"metric": "rouge_l_f1", "max_tokens": 512, "prompt": (
        "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:"
    )},
}


def qa_f1(prediction: str, reference: str) -> float:
    """LongBench's English QA normalization and multiset token F1."""
    def tokens(text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return re.sub(r'\b(a|an|the)\b', ' ', text).split()
    predicted, expected = tokens(prediction), tokens(reference)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    return 2 * overlap / (len(predicted) + len(expected))


def score_prediction(task: str, prediction: str, references: list[str]) -> float:
    if not references or not all(isinstance(value, str) for value in references):
        raise ValueError("LongBench requires a nonempty list of reference answers.")
    if TASKS[task]["metric"] == "token_f1":
        return max(qa_f1(prediction, answer) for answer in references)
    from rouge import Rouge
    scorer = Rouge()
    def score(answer):
        # Matches LongBench: the rouge package rejects empty/invalid summaries.
        try:
            return scorer.get_scores([prediction], [answer], avg=True)["rouge-l"]["f"]
        except (ValueError, RecursionError):
            return 0.0
    return max(score(answer) for answer in references)


def prepare_prompt(tokenizer, task: str, row: dict, budget: int) -> dict:
    """Keep instructions/question intact and trim only the context in the middle."""
    if tokenizer.bos_token_id is None:
        raise ValueError("The tokenizer must define a BOS token.")
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    before, after = TASKS[task]["prompt"].split("{context}")
    prefix, suffix = chat_parts(tokenizer, before.format(input=row.get("input", "")),
                                after.format(input=row.get("input", "")))
    context = encode(row["context"])
    available = budget - len(prefix) - len(suffix)
    if available < 1:
        raise ValueError("Input budget cannot accommodate instructions and question.")
    original_length = len(context)
    if original_length > available:
        left = (available + 1) // 2
        right = available // 2
        context = context[:left] + (context[-right:] if right else [])
    return {
        "prompt_token_ids": prefix + context + suffix,
        "original_context_tokens": original_length,
        "retained_context_tokens": len(context),
        "truncated": original_length > len(context),
    }


def read_task(archive: zipfile.ZipFile, task: str) -> list[dict]:
    """Read the official JSONL directly, without executing its dataset script."""
    with archive.open(f"data/{task}.jsonl") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"LongBench task has no examples: {task}")
    return rows


def summarize_trials(trials: list[dict]) -> dict:
    if not trials:
        raise ValueError("No LongBench trials to score.")
    groups = {}
    bins = {}
    for trial in trials:
        value = trial["score"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("LongBench score must be finite and between zero and one.")
        groups.setdefault(trial["input_budget"], []).append(trial)
        length = trial["input_tokens"]
        label = "<=2048" if length <= 2048 else "2049-4096" if length <= 4096 else ">4096"
        bins.setdefault((trial["input_budget"], label), []).append(trial)
    def summary(rows):
        return {"score": 100 * sum(row["score"] for row in rows) / len(rows),
                "num_examples": len(rows), "truncated_examples": sum(row["truncated"] for row in rows)}
    return {
        "unit": "0-100 task score; higher is better",
        "by_input_budget": [{"input_budget": budget, **summary(rows)}
                            for budget, rows in sorted(groups.items())],
        "by_actual_input_length": [
            {"input_budget": budget, "input_length_bin": label, **summary(rows)}
            for (budget, label), rows in sorted(bins.items())
        ],
    }


def evaluate_long_context(model_path: str) -> dict:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # Cache the original archive; its content checksum identifies the exact data.
    path = hf_hub_download(repo_id=DATASET, filename="data.zip", repo_type="dataset")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    settings = {**ENGINE_SETTINGS, "max_model_len": None}
    engine = LLM(model=model_path, **settings)
    model_limit = int(engine.llm_engine.model_config.max_model_len)
    task_results = {}
    with zipfile.ZipFile(path) as archive:
        for task, config in TASKS.items():
            rows = read_task(archive, task)
            source_examples = len(rows)
            output_tokens = config["max_tokens"]
            budgets = sorted({min(length, model_limit - output_tokens)
                              for length in CONTEXT_LENGTHS})
            if budgets[0] <= 0:
                raise ValueError(f"Model cannot accommodate {task}'s output reservation.")
            params = SamplingParams(temperature=0.0, max_tokens=output_tokens,
                                    stop_token_ids=stop_token_ids(tokenizer))
            trials = []
            for budget in budgets:
                for index, row in enumerate(rows):
                    prepared = prepare_prompt(tokenizer, task, row, budget)
                    tokens = prepared.pop("prompt_token_ids")
                    outputs = engine.generate([TokensPrompt(prompt_token_ids=tokens)], params, use_tqdm=False)
                    if len(outputs) != 1 or list(outputs[0].prompt_token_ids) != tokens:
                        raise ValueError("LongBench input was truncated or returned out of order.")
                    if len(outputs[0].outputs) != 1:
                        raise ValueError("Expected one generated answer per LongBench example.")
                    answer = outputs[0].outputs[0]
                    trials.append({
                        "example_id": row.get("_id", f"{task}-{index}"),
                        "input_budget": budget, "input_tokens": len(tokens), **prepared,
                        "prediction": answer.text, "references": row["answers"],
                        "score": score_prediction(task, answer.text, row["answers"]),
                        "generated_tokens": len(answer.token_ids), "finish_reason": answer.finish_reason,
                    })
            task_results[task] = {
                "metric": config["metric"], "output_tokens": output_tokens,
                "source_examples": source_examples, "selected_examples": len(rows),
                **summarize_trials(trials), "trials": trials,
            }
    return {
        "status": "ok", "benchmark": "LongBench_v1_selected_tasks", "dataset": DATASET,
        "split": "test", "archive_sha256": digest.hexdigest(),
        "dataset_snapshot": Path(path).parent.name,
        "requested_context_lengths": list(CONTEXT_LENGTHS), "model_context_limit": model_limit,
        "engine_settings": {**settings, "max_model_len": model_limit},
        "batch_size": 1, **prompt_protocol(tokenizer),
        "protocol": "official task text in checkpoint user/assistant chat wrapper; context-only token middle truncation; best-reference score",
        "protocol_variant": "input-budget sweep; not an original LongBench leaderboard reproduction",
        "cache_evaluation": {
            "mode": "vllm_default_cache_baseline", "quantized_cache_verified": False,
            "matched_budget_comparison": "blocked: cache backends and memory controls not integrated",
        },
        "tasks": task_results,
    }
