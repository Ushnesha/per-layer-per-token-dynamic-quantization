#!/bin/bash
# AWQ W4A16: quantize on the shared calibration set, then run the full evaluation.
# Needs more host RAM than an interactive 24 GB job (GPTQModel's AWQ caches
# per-layer activations in CPU memory).
#
# Submit from the project root:  sbatch scripts/awq_w4.sbatch
#SBATCH --job-name=awq-w4
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/awq_w4_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./.conda/envs/llm-serv

python run.py quantize --method awq --bits 4
python run.py evaluate --method awq --bits 4
