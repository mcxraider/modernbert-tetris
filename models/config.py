"""Non-secret runtime config, read from config.json at the repo root.

The Node side reads the same file (lib/config.mjs). Keeps model paths out of
.env (which holds only secrets). Env vars still win when set, so a one-off
`MODERNBERT_ONNX=... npm run dev` keeps working.
"""

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG = json.loads((_REPO_ROOT / "config.json").read_text())


def modernbert_onnx():
    return os.environ.get("MODERNBERT_ONNX") or _CONFIG["modernbertOnnx"]


def modernbert_checkpoint_onnx():
    return os.environ.get("MODERNBERT_CHECKPOINT_ONNX") or _CONFIG["modernbertCheckpointOnnx"]
