"""Long-lived ModernBERT inference worker: load once, answer one JSON line per move.

Run from the repo root: `python3 models/serve.py`. Loads the scorer and tokenizer
a single time, then reads one request per stdin line and writes one reply per
stdout line. The server in lib/modernbert.mjs spawns and talks to this process.

Protocol (newline-delimited JSON, UTF-8):
  startup  -> {"ready": true, "source": "onnx"|"checkpoint"|"baseline"}  (or {"ready": false, "error": ...})
  request  <- {"prompts": [...], "candidate_ids": [...]}          (one state's candidates)
  reply    -> {"chosen_candidate_id": ..., "candidate_ids": [...], "probabilities": [...]}
              or {"error": "..."} for a bad request (worker stays up)

MODERNBERT_ONNX serves an exported ONNX graph via ONNX Runtime (see
models/export_onnx.py). Otherwise MODERNBERT_CHECKPOINT selects a trained head;
without either the UNTRAINED baseline (pretrained encoder + random head) runs
and plays near-randomly.
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

        onnx_path = os.environ.get("MODERNBERT_ONNX")
        checkpoint = os.environ.get("MODERNBERT_CHECKPOINT")
        if onnx_path:
            from models.onnx_scorer import OnnxScorer
            model = OnnxScorer(onnx_path)
            source = "onnx"
        elif checkpoint:
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
        # Long-lived torch path: fuse the compute once so every move is cheap.
        # ONNX Runtime already fused the graph at session load, so skip it there.
        if source != "onnx":
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
            # Debug: logits -> softmax per candidate, to the dev console (stderr).
            rows = sorted(zip(result["candidate_ids"], result["logits"], result["probabilities"]),
                          key=lambda r: r[2], reverse=True)
            print(f"[modernbert-{source}] {len(rows)} candidates ({source}), chosen={result['chosen_candidate_id']}",
                  file=sys.stderr)
            for cid, logit, prob in rows:
                print(f"  {str(cid):<6} logit={logit:+.4f}  p={prob:.4f}", file=sys.stderr)
            sys.stderr.flush()
            _emit({
                "chosen_candidate_id": result["chosen_candidate_id"],
                "candidate_ids": result["candidate_ids"],
                "probabilities": result["probabilities"],
            })
        except Exception as error:  # bad request: report and keep serving
            _emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
