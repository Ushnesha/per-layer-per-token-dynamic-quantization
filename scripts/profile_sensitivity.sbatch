#!/bin/bash
# Per-unit sensitivity profile (cheap one-pass signal + end-to-end KL reference)
# for per-layer mixed-precision allocation. ~10 min on one A100.
#
# Submit from the project root:  sbatch scripts/profile_sensitivity.sbatch
#SBATCH --job-name=sens-profile
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/profile_sensitivity_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.conda/envs/llm-serv

python run.py profile
