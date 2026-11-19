"""Full evaluation runtime configuration."""

import os

# CUDA workers must start clean rather than inheriting initialized driver state.
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
PROFILE = 'full'
SCOPE = 'all'
