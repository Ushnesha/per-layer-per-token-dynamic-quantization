#!/bin/bash
# Harder downstream tasks (ARC-Challenge, MMLU; 0-shot, vLLM backend) for every evaluated
# checkpoint. HellaSwag is saturated (all quantized models within its noise), so these
# separate configurations better. Only lm_eval runs; results are merged into the existing
# results/<method>/<label>.json (PPL, memory, latency, HellaSwag are kept). ~10 min each.
#
# Submit from the project root:  sbatch scripts/extra_tasks.sbatch [method:bits[:name] ...]
# or run inside an interactive GPU job:  bash scripts/extra_tasks.sbatch
#SBATCH --job-name=extra-tasks
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/extra_tasks_%j.out

set -uo pipefail
# sbatch: the submit dir. Interactive (OnDemand sets SLURM_SUBMIT_DIR to its dashboard): the repo of this script.
{ cd "${SLURM_SUBMIT_DIR:-.}" 2>/dev/null && [ -f run.py ]; } || cd "$(dirname "$0")/.."
PY=./.conda/envs/llm-serv/bin/python
TASKS=(arc_challenge mmlu)

ENTRIES=("$@")
if [ ${#ENTRIES[@]} -eq 0 ]; then
    ENTRIES=(fp16:16 gptq:8 smoothquant:8 awq:4 gptq:4 mixed:4:lm8-down8)
    for b in 5.40 5.50 5.75 6.00 6.50 7.00; do ENTRIES+=("mixed:4:ref-kl-$b"); done
    for n in pos-5.75 rand0-5.75 rand1-5.75 che-abs-5.75 pos-5.50 rand0-5.50 che-abs-5.50; do ENTRIES+=("mixed:4:$n"); done
fi

for entry in "${ENTRIES[@]}"; do
    IFS=: read -r method bits name <<< "$entry"
    tag="${method}-${bits}${name:+-$name}"
    echo "[$(date +%T)] ${tag}: ${TASKS[*]}"
    if $PY run.py evaluate --method "$method" --bits "$bits" ${name:+--name "$name"} \
           --skip-ppl --skip-memory --skip-latency --tasks "${TASKS[@]}" > "logs/extra_tasks_${tag}.log" 2>&1 \
       && ! grep -q "lm_eval failed\|Model not found" "logs/extra_tasks_${tag}.log"; then
        echo "[$(date +%T)] ${tag} done"
    else
        echo "[$(date +%T)] ${tag} FAILED (see logs/extra_tasks_${tag}.log)"
    fi
done
