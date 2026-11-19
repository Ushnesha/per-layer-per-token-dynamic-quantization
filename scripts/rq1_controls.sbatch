#!/bin/bash
# RQ1 controls: separate the per-layer allocation's gain from the 8-bit lm_head, and
# test the allocation on top of AWQ. A failed step is logged and skipped.
#
#   mixed-awq w4        AWQ W4 repacked as compressed-tensors (builds the AWQ base;
#                       must reproduce the AWQ baseline's PPL; kernel-parity control)
#   mixed-awq lm8       AWQ W4 + 8-bit lm_head
#   mixed lm8           GPTQ W4 + 8-bit lm_head
#   mixed-awq ref-kl-*  the sensitivity-guided allocations of the GPTQ sweep, on AWQ
#
# Submit from the project root:  sbatch scripts/rq1_controls.sbatch [budget ...]
# or run inside an interactive GPU job:  bash scripts/rq1_controls.sbatch 5.5 5.75
# AWQ needs more host RAM than a default 24 GB interactive job (it gets OOM-killed).
# Steps whose results JSON already exists are skipped, so the script can be rerun.
#SBATCH --job-name=rq1-controls
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rq1_controls_%j.out

set -uo pipefail
{ cd "${SLURM_SUBMIT_DIR:-.}" 2>/dev/null && [ -f run.py ]; } || cd "$(dirname "$0")/.."
PY=./.conda/envs/llm-serv/bin/python

BUDGETS=("$@")
[ ${#BUDGETS[@]} -eq 0 ] && BUDGETS=(5.4 5.5 5.75 6.0 6.5 7.0)

# step <method> <name> <quantize args...>
step() {
    local method=$1 name=$2; shift 2
    local log="logs/${method}_${name}"
    local label=$([ "$method" = mixed ] && echo "mixed-${name}" || echo "mixed-awq-${name}")
    local result="results/${method}/${label}.json"
    # evaluate exits 0 even when vLLM fails, so success means a PPL in the results file
    if grep -q '"wikitext"' "$result" 2>/dev/null; then
        echo "[$(date +%T)] ${method} ${name} already evaluated, skipping"
        return
    fi
    echo "[$(date +%T)] ${method} ${name}"
    if $PY run.py quantize --method "$method" --bits 4 --name "$name" "$@" > "${log}_quantize.log" 2>&1 \
       && $PY run.py evaluate --method "$method" --bits 4 --name "$name" > "${log}_evaluate.log" 2>&1 \
       && grep -q '"wikitext"' "$result"; then
        echo "[$(date +%T)] ${method} ${name} done"
    else
        echo "[$(date +%T)] ${method} ${name} FAILED (see ${log}_*.log)"
    fi
}

step mixed-awq w4
step mixed-awq lm8 --assign lm_head=8
step mixed lm8 --assign lm_head=8
for b in "${BUDGETS[@]}"; do
    step mixed-awq "$(printf "ref-kl-%.2f" "$b")" --budget "$b"
done
