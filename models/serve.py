"""Long-lived ModernBERT inference worker: load once, answer one JSON line per move.

Run from the repo root: `python3 models/serve.py`. Loads the scorer and tokenizer
a single time, then reads one request per stdin line and writes one reply per
stdout line. The server in lib/modernbert.mjs spawns and talks to this process.

Protocol (newline-delimited JSON, UTF-8):
  startup  -> {"ready": true, "source": "checkpoint"|"baseline"}  (or {"ready": false, "error": ...})
  request  <- {"prompts": [...], "candidate_ids": [...]}          (one state's candidates)
  reply    -> {"chosen_candidate_id": ..., "candidate_ids": [...], "probabilities": [...]}
              or {"error": "..."} for a bad request (worker stays up)

MODERNBERT_CHECKPOINT selects a trained head; without it the UNTRAINED baseline
(pretrained encoder + random head) runs and plays near-randomly.
"""

import json
import os
import sys
from pathlib import Path

# `models.inference` imports `models.modernbert`, so the repo root must be importable
# even when launched as `python3 models/serve.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _emit(obj):
    sys.stdout.write(json.dumps(obj, allow_nan=False) + "\n")
    sys.stdout.flush()


def main():
    try:
        import torch
        from transformers import AutoTokenizer

        from models.inference import run_inference
        from models.modernbert import ModernBertScorer

        checkpoint = os.environ.get("MODERNBERT_CHECKPOINT")
        if checkpoint:
            model = ModernBertScorer.load_checkpoint(Path(checkpoint))
            source = "checkpoint"
        else:
            torch.manual_seed(7)  # reproducible random head for the baseline
            model = ModernBertScorer.from_pretrained()
            source = "baseline"
        tokenizer_id = model.tokenizer_id
        if not tokenizer_id:
            raise ValueError("Checkpoint has no tokenizer identity; set MODERNBERT_CHECKPOINT to one with metadata")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side="right")
        # Long-lived: fuse the compute once so every subsequent move is cheap.
        model.compile_scorer()
    except Exception as error:  # torch missing, download failure, bad checkpoint
        _emit({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return

    _emit({"ready": True, "source": source})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            result = run_inference(model, tokenizer, req.get("prompts"), req.get("candidate_ids"))
            _emit({
                "chosen_candidate_id": result["chosen_candidate_id"],
                "candidate_ids": result["candidate_ids"],
                "probabilities": result["probabilities"],
            })
        except Exception as error:  # bad request: report and keep serving
            _emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
