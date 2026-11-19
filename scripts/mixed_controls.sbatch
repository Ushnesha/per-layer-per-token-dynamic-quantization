#!/bin/bash
# Controls for sensitivity-guided allocation: same budgets, lm_head at 8 bits, but the
# decoder units are chosen randomly, by position (outermost layers first), or by the
# cheap one-pass signal. ~30 min per run on one A100. A failed run is logged and skipped.
#
# Submit from the project root:  sbatch scripts/mixed_controls.sbatch [signal:seed:budget ...]
# or run inside an interactive GPU job:  bash scripts/mixed_controls.sbatch
#SBATCH --job-name=mixed-controls
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mixed_controls_%j.out

set -uo pipefail
# sbatch: the submit dir. Interactive (OnDemand sets SLURM_SUBMIT_DIR to its dashboard): the repo of this script.
{ cd "${SLURM_SUBMIT_DIR:-.}" 2>/dev/null && [ -f run.py ]; } || cd "$(dirname "$0")/.."
PY=./.conda/envs/llm-serv/bin/python

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(position:0:5.75 random:0:5.75 random:1:5.75 cheap:0:5.75
                                position:0:5.5 random:0:5.5 cheap:0:5.5)

for run in "${RUNS[@]}"; do
    IFS=: read -r signal seed b <<< "$run"
    case "$signal" in
        random)   name=$(printf "rand%d-%.2f" "$seed" "$b"); args=(--signal random --seed "$seed") ;;
        position) name=$(printf "pos-%.2f" "$b");            args=(--signal position) ;;
        cheap)    name=$(printf "che-abs-%.2f" "$b");        args=(--signal cheap --metric abs) ;;
        *)        echo "unknown signal: $signal"; continue ;;
    esac
    echo "[$(date +%T)] ${signal} seed=${seed} budget ${b} GB -> ${name}"
    if $PY run.py quantize --method mixed --bits 4 --budget "$b" --name "$name" "${args[@]}" \
           > "logs/mixed_${name}_quantize.log" 2>&1 \
       && $PY run.py evaluate --method mixed --bits 4 --name "$name" > "logs/mixed_${name}_evaluate.log" 2>&1; then
        echo "[$(date +%T)] ${name} done"
    else
        echo "[$(date +%T)] ${name} FAILED (see logs/mixed_${name}_*.log)"
    fi
done
