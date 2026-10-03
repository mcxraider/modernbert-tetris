"""Export a ModernBertScorer to ONNX for CPU serving via ONNX Runtime.

Exports the guard-free compute (_score: encoder -> CLS -> head -> [N] logits),
NOT forward() whose validation branches are Python control flow the tracer
can't follow. Softmax/argmax stay in Python (run_inference), so the graph is
just the heavy part. Both batch (dim 0) and sequence length (dim 1) are dynamic.

Writes model.onnx plus a sidecar model.onnx.json holding the tokenizer id and
the two config fields run_inference needs, so the ONNX artifact serves without
the original .pt. The exporter stores weights as external data in a sibling
model.onnx.data file (~569 MB for ModernBERT-base); ORT loads it automatically
from the same directory, so keep the three files together. The export is
self-checked against torch before success.

  python -m models.export_onnx --baseline --tokenizer answerdotai/ModernBERT-base
  python -m models.export_onnx --checkpoint scorer.pt
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from models.modernbert import ModernBertScorer


class _ScoreGraph(nn.Module):
    """Thin wrapper so torch.onnx traces _score, bypassing forward's guards."""

    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, input_ids, attention_mask):
        return self.scorer._score(input_ids, attention_mask)


def export(model, out_path):
    """Export `model` (an eval-mode ModernBertScorer) to `out_path` and self-check."""
    model.eval()
    config = model.encoder.config
    graph = _ScoreGraph(model).eval()
    # A tiny but realistic batch: 2 candidates, short sequence, CLS first.
    seq = 8
    input_ids = torch.full((2, seq), config.cls_token_id, dtype=torch.int64)
    input_ids[:, 1:] = torch.randint(0, config.vocab_size, (2, seq - 1))
    attention_mask = torch.ones((2, seq), dtype=torch.int64)
    dynamic_axes = {
        "input_ids": {0: "candidates", 1: "tokens"},
        "attention_mask": {0: "candidates", 1: "tokens"},
        "logits": {0: "candidates"},
    }
    with torch.inference_mode():
        try:
            torch.onnx.export(
                graph, (input_ids, attention_mask), str(out_path),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes=dynamic_axes, dynamo=True)
        except Exception as dynamo_error:  # fall back to the legacy tracer
            print(f"dynamo export failed ({dynamo_error}); retrying with the legacy tracer")
            torch.onnx.export(
                graph, (input_ids, attention_mask), str(out_path),
                input_names=["input_ids", "attention_mask"], output_names=["logits"],
                dynamic_axes=dynamic_axes, opset_version=17)
        expected = model._score(input_ids, attention_mask)

    import onnxruntime as ort

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    (actual,) = session.run(None, {
        "input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-3, atol=1e-4)

    sidecar = {
        "tokenizer_id": model.tokenizer_id,
        "cls_token_id": config.cls_token_id,
        "max_position_embeddings": config.max_position_embeddings,
    }
    Path(str(out_path) + ".json").write_text(json.dumps(sidecar, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Saved scorer including its learned head")
    source.add_argument("--baseline", action="store_true", help="Pretrained encoder + random head")
    parser.add_argument("--tokenizer", help="Tokenizer ID; overrides/sets checkpoint metadata")
    parser.add_argument("--seed", type=int, default=7, help="Random-head seed for --baseline")
    parser.add_argument("-o", "--out", type=Path, default=Path("models/onnx/model.onnx"),
                        help="Output .onnx path (sibling .onnx.data + .onnx.json land beside it)")
    args = parser.parse_args()
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        model = (ModernBertScorer.from_pretrained() if args.baseline
                 else ModernBertScorer.load_checkpoint(args.checkpoint))
        if args.tokenizer:
            model.tokenizer_id = args.tokenizer
        if not model.tokenizer_id:
            raise ValueError("Specify --tokenizer; checkpoint has no tokenizer identity")
        export(model, args.out)
        print(f"wrote {args.out} and {args.out}.json (tokenizer: {model.tokenizer_id})")
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
