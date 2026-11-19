#!/bin/bash
# Run from the repository root:
#   sbatch scripts/evaluation.sbatch /absolute/path/to/model
# A compatible conda-pack archive must already be prepared; see EVALUATION_SOL.md.
#SBATCH --job-name=model-evaluation
#SBATCH --account=grp_jzou22
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=evaluation_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
model_path="${1:?Usage: sbatch scripts/evaluation.sbatch /path/to/model}"
shift
if [[ ! -f "$model_path/config.json" ]]; then
    echo "Model must be a local checkpoint directory containing config.json: $model_path" >&2
    exit 1
fi
mkdir -p logs

# source code and reports stay in the home-directory repository.
module load mamba/latest

evaluation_storage="/scratch/$USER/benchmark-evaluation"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$evaluation_storage/cache/huggingface/hub"
export HF_DATASETS_CACHE="$evaluation_storage/cache/huggingface/datasets"
export HF_XET_CACHE="$evaluation_storage/cache/huggingface/xet"
export CONDA_PKGS_DIRS="$evaluation_storage/cache/conda"
export PIP_CACHE_DIR="$evaluation_storage/cache/pip"
export TORCH_HOME="$evaluation_storage/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$evaluation_storage/cache/torchinductor"
export TRITON_CACHE_DIR="$evaluation_storage/cache/triton"
export VLLM_CACHE_ROOT="$evaluation_storage/cache/vllm"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$evaluation_storage/cache" "$evaluation_storage/tmp"
export TMPDIR="$evaluation_storage/tmp"


# Discover this user's prepared archive; an explicit path can point to a shared archive.
if [[ -z "${ADITYA_ENV_ARCHIVE:-}" ]]; then
    if [[ -s "$evaluation_storage/archives/llm-serv.tar" ]]; then
        export ADITYA_ENV_ARCHIVE="$evaluation_storage/archives/llm-serv.tar"
    else
        ADITYA_ENV_ARCHIVE="$(ls -t "$evaluation_storage"/archives/llm-serv-*.tar 2>/dev/null | head -n 1 || true)"
        export ADITYA_ENV_ARCHIVE
    fi
fi
if [[ ! -s "${ADITYA_ENV_ARCHIVE:-}" ]]; then
    echo "A prepared evaluation environment archive is required. See scripts/EVALUATION_SOL.md." >&2
    exit 1
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Profiler availability is recorded by each metric; absence does not prevent core evaluation.
module load cuda-12.9.0-gcc-12.1.0 || echo "CUDA profiler module unavailable; profiling may be unsupported." >&2

: "${SLURM_JOB_ID:?Run inside a Slurm allocation}"
: "${ADITYA_ENV_ARCHIVE:?Set the packed environment archive path}"
test -s "$ADITYA_ENV_ARCHIVE" || { echo "Missing environment archive" >&2; exit 1; }

# /tmp is disk-backed on the inspected Sol compute node. Verify every allocation
# rather than silently staging onto tmpfs or another shared filesystem.
local_fs_type="$(findmnt -n -o FSTYPE -T /tmp)"
case "$local_fs_type" in
    xfs|ext4) ;;
    *) echo "Expected node-local disk at /tmp, found $local_fs_type" >&2; exit 1 ;;
esac
archive_bytes="$(stat -c %s "$ADITYA_ENV_ARCHIVE")"
available_bytes="$(df -B1 --output=avail /tmp | tail -n 1 | tr -d ' ')"
# Uncompressed tar bounds payload size; leave room for prefix rewriting/caches.
required_bytes=$((archive_bytes + archive_bytes / 5 + 5368709120))
if (( available_bytes < required_bytes )); then
    echo "Insufficient local disk: need $required_bytes bytes, have $available_bytes" >&2
    exit 1
fi

evaluation_local_root="$(mktemp -d "/tmp/ae-${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf -- "$evaluation_local_root"' EXIT
local_environment="$evaluation_local_root/env"
mkdir -p "$local_environment" "$evaluation_local_root/tmp"
echo "$(date -Is) Unpacking environment to $local_environment"
tar -xf "$ADITYA_ENV_ARCHIVE" -C "$local_environment"
export TMPDIR="$evaluation_local_root/tmp"
export PATH="$local_environment/bin:$PATH"
export CONDA_PREFIX="$local_environment"
unset PYTHONHOME PYTHONPATH
"$local_environment/bin/python" "$local_environment/bin/conda-unpack"

# Check both interpreter relocation and origins without importing GPU libraries.
"$local_environment/bin/python" - <<'PY'
import importlib.util
import os
from pathlib import Path
import sys
prefix = Path(os.environ['CONDA_PREFIX']).resolve()
assert Path(sys.prefix).resolve() == prefix, (sys.prefix, prefix)
for name in ('torch', 'transformers', 'vllm', 'sympy', 'scipy', 'sklearn'):
    spec = importlib.util.find_spec(name)
    if spec is None or not Path(spec.origin).resolve().is_relative_to(prefix):
        raise RuntimeError(f'{name} does not resolve inside the local environment: {spec}')
print('Local interpreter and package paths verified:', sys.executable, flush=True)
PY
echo "$(date -Is) Local environment ready"

python -u -m src.evaluation.benchmark.evaluate --model "$model_path" "$@"
