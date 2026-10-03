"""Batch all candidate prompts for ONE state, then choose via softmax."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.modernbert import ModernBertScorer, candidate_probabilities


def run_inference(model, tokenizer, prompts, candidate_ids=None, valid_mask=None):
    """Return JSON-ready logits, probabilities and the chosen candidate prompt.

    Prompts use public/prompt.js's STATE\n...\n\nCANDIDATE\n... format.
    Input order is preserved. Ties choose the first eligible candidate.
    The model is put in eval mode and left there; no gradients are recorded.
    """
    if not isinstance(prompts, (list, tuple)) or not prompts:
        raise ValueError("Provide a nonempty list of candidate prompts for one state")
    states = []
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.startswith("STATE\n"):
            raise ValueError("Each prompt must start with STATE followed by a newline")
        state, separator, candidate = prompt.partition("\n\nCANDIDATE\n")
        if not separator or not candidate.strip() or not state[6:].strip():
            raise ValueError("Each prompt must contain nonempty STATE and CANDIDATE blocks")
        states.append(state)
    if len(set(states)) != 1:
        raise ValueError("All prompts must describe the same current state")
    count = len(prompts)
    if candidate_ids is None:
        candidate_ids = list(range(count))
    if (not isinstance(candidate_ids, (list, tuple)) or len(candidate_ids) != count
            or any(type(i) not in (str, int) or i == "" for i in candidate_ids)
            or len(set(candidate_ids)) != count):
        raise ValueError("candidate_ids must be unique strings or integers, one per prompt")
    if valid_mask is not None:
        if (not isinstance(valid_mask, (list, tuple)) or len(valid_mask) != count
                or any(type(v) is not bool for v in valid_mask) or not any(valid_mask)):
            raise ValueError("valid_mask must contain one Boolean per prompt and allow a candidate")
    if tokenizer.padding_side != "right":
        raise ValueError("Tokenizer must use right padding")
    if tokenizer.cls_token_id != model.encoder.config.cls_token_id:
        raise ValueError("Tokenizer CLS token does not match the model")

    # Check lengths before padding, and never silently drop board information.
    encoded = tokenizer(list(prompts), add_special_tokens=True, truncation=False,
                        padding=False, return_attention_mask=True)
    if max(map(len, encoded["input_ids"])) > model.encoder.config.max_position_embeddings:
        raise ValueError("Prompt exceeds model context length; shorten it explicitly")
    batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        mask = None if valid_mask is None else torch.tensor(valid_mask, device=device)
        probabilities = candidate_probabilities(logits, [count], mask)
        chosen = int(probabilities.argmax().item())
    return {
        "candidate_ids": list(candidate_ids),
        "logits": logits.float().cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
        "chosen_index": chosen,
        "chosen_candidate_id": candidate_ids[chosen],
        "chosen_prompt": prompts[chosen],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON keyed by state/candidate ID (also accepts legacy arrays)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Saved scorer including its learned head")
    source.add_argument("--baseline", action="store_true", help="Pretrained encoder + random head")
    parser.add_argument("--state-id", help="Select a state from a multi-state dataset")
    parser.add_argument("--tokenizer", help="Tokenizer ID or directory; defaults to checkpoint metadata")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=7, help="Random-head seed for --baseline")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the scorer; worth it only across many calls, not one-shot")
    args = parser.parse_args()
    try:
        rows = json.loads(args.input.read_text())
        if isinstance(rows, dict):
            if not rows:
                raise ValueError("Dataset contains no states")
            if args.state_id is None and len(rows) != 1:
                raise ValueError("Dataset contains multiple states; select --state-id")
            state_id = args.state_id if args.state_id is not None else next(iter(rows))
            if state_id not in rows:
                raise ValueError("No candidates match --state-id")
            candidates = rows[state_id]
            if not isinstance(candidates, dict) or not candidates:
                raise ValueError("Each state must map candidate IDs to nonempty candidate objects")
            rows = []
            for candidate_id, candidate in candidates.items():
                if not isinstance(candidate, dict):
                    raise ValueError("Each candidate must be an object")
                if (candidate.get("state_id", state_id) != state_id
                        or candidate.get("candidate_id", candidate_id) != candidate_id):
                    raise ValueError("Candidate metadata conflicts with its state/candidate keys")
                rows.append({**candidate, "state_id": state_id, "candidate_id": candidate_id})
        if not isinstance(rows, list) or not rows:
            raise ValueError("Input must be a nonempty JSON array")
        if all(isinstance(row, dict) for row in rows):
            if args.state_id is not None:
                rows = [row for row in rows if row.get("state_id") == args.state_id]
            if not rows:
                raise ValueError("No candidates match --state-id")
            if any(not isinstance(row.get("state_id"), str) for row in rows):
                raise ValueError("Dataset rows require a string state_id")
            if len({row["state_id"] for row in rows}) != 1:
                raise ValueError("Dataset contains multiple states; select --state-id")
            prompts = [row.get("prompt") for row in rows]
            ids = [row.get("candidate_id") for row in rows]
        elif all(isinstance(row, str) for row in rows):
            if args.state_id is not None:
                raise ValueError("--state-id requires dataset rows, not plain strings")
            prompts, ids = rows, None
        else:
            raise ValueError("Input must contain only prompts or only dataset rows")
        torch.manual_seed(args.seed)
        model = (ModernBertScorer.from_pretrained() if args.baseline
                 else ModernBertScorer.load_checkpoint(args.checkpoint))
        tokenizer_id = args.tokenizer or model.tokenizer_id
        if not tokenizer_id:
            raise ValueError("Specify --tokenizer; checkpoint has no tokenizer identity")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, padding_side="right")
        model = model.to(args.device)
        if args.compile:
            model.compile_scorer()
        result = run_inference(model, tokenizer, prompts, ids)
        if ids is not None:
            result["state_id"] = rows[0]["state_id"]
        result["source"] = "pretrained_encoder_random_head" if args.baseline else "checkpoint"
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
