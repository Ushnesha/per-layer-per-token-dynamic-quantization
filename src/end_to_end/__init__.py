"""End-to-end pipeline: per-layer weight quantization (src/adaptive) + per-token KV-cache
quantization (src/per_token_quant), evaluated together.

Run everything from the repo root, e.g. `python -m src.end_to_end.check_weights`.
"""

import sys
from pathlib import Path

# The per-token code imports its own folders as top-level packages (`oracle`, `prototype`,
# `l2_norm_mlp`), so put src/per_token_quant on the import path.
_PER_TOKEN_QUANT = Path(__file__).resolve().parent.parent / "per_token_quant"
if str(_PER_TOKEN_QUANT) not in sys.path:
    sys.path.insert(0, str(_PER_TOKEN_QUANT))