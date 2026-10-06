"""Long-lived ModernBERT inference worker: load once, answer one JSON line per move.

Run from the repo root: `python3 models/serve.py`. Loads the scorer and tokenizer
a single time, then reads one request per stdin line and writes one reply per
stdout line. The server in lib/modernbert.mjs spawns and talks to this process.

Protocol (newline-delimited JSON, UTF-8):
  startup  -> {"ready": true, "source": "onnx"|"checkpoint"|"baseline"}  (or {"ready": false, "error": ...})
  request  <- {"prompts": [...], "candidate_ids": [...]}          (one state's candidates)
  reply    -> {"chosen_candidate_id": ..., "candidate_ids": [...], "probabilities": [...]}
              or {"error": "..."} for a bad request (worker stays up)

config.json's modernbertOnnx serves an exported ONNX graph via ONNX Runtime
(see models/export_onnx.py). Otherwise MODERNBERT_CHECKPOINT selects a trained
head; without either the UNTRAINED baseline (pretrained encoder + random head)
runs and plays near-randomly. Paths live in config.json (models/config.py).
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


def load_model(model_id):
    """Build one (model, tokenizer, source) triple for a model id.

    "base"       -> config.json modernbertOnnx, else MODERNBERT_CHECKPOINT, else untrained baseline.
    "checkpoint" -> the exported checkpoint ONNX (config.json modernbertCheckpointOnnx).

    Paths come from models/config.py (config.json; env vars override).
    """
    import torch
    from transformers import AutoTokenizer

    from models.modernbert import ModernBertScorer

    from models.config import modernbert_checkpoint_onnx, modernbert_onnx

    if model_id == "checkpoint":
        from models.onnx_scorer import OnnxScorer
        model = OnnxScorer(modernbert_checkpoint_onnx())
        source = "checkpoint"
    else:  # "base"
        onnx_path = modernbert_onnx()
        checkpoint = os.environ.get("MODERNBERT_CHECKPOINT")
        if onnx_path:
            from models.onnx_scorer import OnnxScorer
            model = OnnxScorer(onnx_path)
            source = "onnx"
        elif checkpoint:
            model = ModernBertScorer.load_checkpoint(Path(checkpoint))
            source = "checkpoint-pt"
        else:
            torch.manual_seed(7)  # reproducible random head for the baseline
            model = ModernBertScorer.from_pretrained()
            source = "baseline"
    tokenizer_id = model.tokenizer_id
    if not tokenizer_id:
        raise ValueError("Model has no tokenizer identity; export/checkpoint it with metadata")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side="right")
    # Long-lived torch path: fuse the compute once so every move is cheap.
    # ONNX Runtime already fused the graph at load, and OnnxScorer has no
    # compile_scorer — so only the torch models (checkpoint-pt, baseline) compile.
    if hasattr(model, "compile_scorer"):
        model.compile_scorer()
    return model, tokenizer, source


def main():
    import gc
    current = {}  # single slot: {model_id: (model, tokenizer, source)}. One model resident at a time.

    def get_model(model_id):
        if model_id in current:
            return current[model_id]
        current.clear()  # evict the other model before loading, so both never co-reside
        gc.collect()
        current[model_id] = load_model(model_id)
        return current[model_id]

    try:
        from models.inference import run_inference
        base = get_model("base")  # eager: readiness signal + fast first move
    except Exception as error:  # torch missing, download failure, bad checkpoint
        _emit({"ready": False, "error": f"{type(error).__name__}: {error}"})
        return

    _emit({"ready": True, "source": base[2]})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            model_id = req.get("model") or "base"
            model, tokenizer, source = get_model(model_id)
            result = run_inference(model, tokenizer, req.get("prompts"), req.get("candidate_ids"))
            # Debug: logits -> softmax per candidate, to the dev console (stderr).
            rows = sorted(zip(result["candidate_ids"], result["logits"], result["probabilities"]),
                          key=lambda r: r[2], reverse=True)
            print(f"[modernbert-{model_id}] {len(rows)} candidates ({source}), chosen={result['chosen_candidate_id']}",
                  file=sys.stderr)
            for cid, logit, prob in rows:
                print(f"  {str(cid):<6} logit={logit:+.4f}  p={prob:.4f}", file=sys.stderr)
            sys.stderr.flush()
            _emit({
                "chosen_candidate_id": result["chosen_candidate_id"],
                "candidate_ids": result["candidate_ids"],
                "probabilities": result["probabilities"],
            })
        except Exception as error:  # bad request / failed lazy load: report and keep serving
            _emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
