"""One LongBench test of prototype v0: HotpotQA, 7,168-token prompts, 50 examples."""
import os
import sys
import zipfile

import torch
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(".."))  # src/, so Aditya's evaluation folder can be imported

from evaluation.benchmark.long_context import DATASET, TASKS, prepare_prompt, read_task, score_prediction
from evaluation.benchmark.instruction_prompt import stop_token_ids
from prototype.adaptive_kv import AdaptiveKVCache, KeyNormLinearScorer, PositionScorer, RandomScorer, generate

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
CKPT = "l2_norm_mlp/results/key_norm_linear_llama3_8b_instruct_nopos.pt"
TASK = "hotpotqa"
INPUT_BUDGET = 7168
NUM_EXAMPLES = 50
LOW_BITS, GROUP_SIZE = 2, None  # the passkey setting: INT2 with group 128
BUDGET = 0.1

tokenizer = AutoTokenizer.from_pretrained(MODEL)
with zipfile.ZipFile(hf_hub_download(repo_id=DATASET, filename="data.zip", repo_type="dataset")) as archive:
    rows = read_task(archive, TASK)[:NUM_EXAMPLES]
prompts = [prepare_prompt(tokenizer, TASK, row, INPUT_BUDGET)["prompt_token_ids"] for row in rows]
print(f"{len(prompts)} {TASK} examples, average prompt {sum(map(len, prompts)) / len(prompts):.0f} tokens")

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

settings = {
    "full precision": (lambda: PositionScorer(), 1.0),
    "uniform INT2": (lambda: PositionScorer(), 0.0),
    f"key norm @ {BUDGET:.0%}": (lambda: KeyNormLinearScorer(CKPT), BUDGET),
    f"position @ {BUDGET:.0%}": (lambda: PositionScorer(), BUDGET),
    f"random @ {BUDGET:.0%}": (lambda: RandomScorer(0), BUDGET),
}
for name, (make_scorer, budget) in settings.items():
    scorer = make_scorer()
    make_cache = lambda: AdaptiveKVCache(scorer, budget=budget, low_bits=LOW_BITS,
                                         group_size=GROUP_SIZE, config=model.config)
    scores = []
    for prompt, row in zip(tqdm(prompts, desc=name), rows):
        out = generate(model, tokenizer, prompt, make_cache, max_new_tokens=TASKS[TASK]["max_tokens"],
                       stop_token_ids=stop_token_ids(tokenizer))
        scores.append(score_prediction(TASK, out["text"], row["answers"]))
    print(f"{name:20s} F1 {100 * sum(scores) / len(scores):.1f}")