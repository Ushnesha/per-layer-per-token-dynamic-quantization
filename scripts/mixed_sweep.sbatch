#!/bin/bash
# Sensitivity-guided mixed precision: for each weight-storage budget (GB), solve the
# allocation from the sensitivity profile, quantize (GPTQ + RTN lm_head), and run the
# full evaluation. ~30 min per budget on one A100. A failed budget is logged and skipped.
#
# Submit from the project root:  sbatch scripts/mixed_sweep.sbatch [budget ...]
# or run inside an interactive GPU job:  bash scripts/mixed_sweep.sbatch 5.5 6.0
#
# Checkpoints (~6-7 GB each) go to models/mixed, a symlink to /scratch (the /home quota is too small).
#SBATCH --job-name=mixed-sweep
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mixed_sweep_%j.out

set -uo pipefail
# sbatch: the submit dir. Interactive (OnDemand sets SLURM_SUBMIT_DIR to its dashboard): the repo of this script.
{ cd "${SLURM_SUBMIT_DIR:-.}" 2>/dev/null && [ -f run.py ]; } || cd "$(dirname "$0")/.."
PY=./.conda/envs/llm-serv/bin/python

BUDGETS=("$@")
[ ${#BUDGETS[@]} -eq 0 ] && BUDGETS=(5.4 5.5 5.75 6.0 6.5 7.0)

for b in "${BUDGETS[@]}"; do
    name=$(printf "ref-kl-%.2f" "$b")
    echo "[$(date +%T)] budget ${b} GB -> ${name}"
    if $PY run.py quantize --method mixed --bits 4 --budget "$b" > "logs/mixed_${name}_quantize.log" 2>&1 \
       && $PY run.py evaluate --method mixed --bits 4 --name "$name" > "logs/mixed_${name}_evaluate.log" 2>&1; then
        echo "[$(date +%T)] ${name} done"
    else
        echo "[$(date +%T)] ${name} FAILED (see logs/mixed_${name}_*.log)"
    fi
done
