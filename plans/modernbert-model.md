# ModernBERT candidate scorer implementation plan

Status: implementation authorized in `/Users/Jerry_YANG_from.TP/govtech/jev-tetris`. This plan was saved before implementation.

Repository findings: `public/prompt.js` already renders the exact input format. Its `stateRows` replaces missing teacher probabilities with zero, so downstream validation must reject all-zero or incomplete distributions. The headless generator does not currently persist run IDs; adding reliable episode metadata before dataset splitting is deferred to the training pipeline. No existing Python model package or dependency manifest exists.

Concrete design: `models/modernbert.py` contains an nn.Module scorer, grouped probability/loss functions, and explicit save/load checkpoint methods using state_dict and ModernBertConfig. Add `requirements-model.txt` and `test/test_modernbert.py`; document Python usage in README. The tensor boundary requires right-padded inputs with CLS first and a binary attention mask. The loss normalizes only sums within 1e-4 of one and rejects larger discrepancies. Group completeness itself must be checked against the teacher candidate list by the future dataset loader.

## Scope

Implement the PyTorch model, candidate-group probability/loss operations, checkpoint persistence, and runnable checks. Do not implement an inference service, arena opponent, or training CLI in this phase. No fine-tuning run is requested yet.

## 1. Inspect the repository and data contract

Read applicable AGENTS.md files, inspect the working tree, and preserve existing changes. Trace `public/prompt.js` and `tools/gen-dataset.mjs` to confirm exactly how state/candidate text and teacher probabilities are produced. Read the relevant player interface for future integration without modifying it. Reuse existing Python packaging if available; the currently inspected project is primarily Node.js and has no visible Python dependency manifest.

Validate the assumption that probabilities within each state are a normalized distribution over the complete candidate set. State IDs and candidate IDs are metadata, not encoder inputs. Keep run IDs for grouped dataset splits. Do not interpret zero teacher probability as an illegal move.

## 2. Minimal files

Use the repository's existing conventions where possible. If there is no Python model package, use `models/modernbert.py`, a small Python dependency manifest, and one runnable test file. Add usage and checkpoint notes to the existing README. Save this document as `plans/modernbert-model.md` first. Avoid introducing a service, framework, plugin system, or training orchestration.

## 3. Architecture

Use Hugging Face's pretrained `answerdotai/ModernBERT-base` encoder through PyTorch. Read hidden width from its configuration. For each tokenized STATE + CANDIDATE prompt, take the final hidden state of the CLS token, apply dropout with probability 0.1, and a bias-free Linear(H, 1). Return one unrestricted scalar per input row. Do not apply sigmoid, ReLU, or independent probability regression.

The module owns the encoder and head, and both remain trainable. Provide an explicit pretrained construction path and support construction from a small ModernBERT configuration for offline tests. Avoid silently falling back to random encoder weights if pretrained loading fails. Use standard attention supported by the installed runtime rather than requiring FlashAttention.

## 4. Tensor interface

The forward method accepts input_ids and attention_mask with shape [N, L], where N is the total number of candidates in a batch, and returns logits with shape [N]. Candidate grouping remains explicit through group sizes or equivalent offsets. Require positive group sizes whose sum equals N. Validate token/mask dimensions and preserve the first CLS position with right padding.

A grouped operation converts logits into separate per-state distributions. All candidates use shared parameters. A Boolean validity mask excludes only candidates explicitly marked invalid; scalar zero remains a valid score. Reject any state with no valid candidate before softmax. Do not pad nonexistent candidates into the encoder batch.

## 5. Distillation objective

For each state, compute log_softmax over its valid candidate logits in float32, and soft-target cross-entropy: L = -sum(q_i * log(p_i)). Average state losses equally. Default temperature is one; no temperature tuning is needed in this phase.

Validate target shape, finiteness, nonnegativity, positive mass, and per-state normalization within a documented rounding tolerance. Invalid candidates must have zero target mass. Avoid 0 multiplied by negative infinity by excluding masked positions from the loss. Retain valid candidates with zero targets in the softmax denominator. Reject incomplete distributions rather than silently treating omitted candidates as zero.

The [0.94, 0.06] example is matched by a logit difference log(0.94/0.06), approximately 2.752. Absolute logit offsets do not change the distribution.

## 6. Tokenization contract and checkpoints

Use the matching pretrained tokenizer with its special tokens, right padding, and explicit length checks. Preserve the existing prompt text. Do not silently truncate the board or candidate description. A full tokenizer/inference wrapper is deferred, but document this caller contract now.

Use standard state_dict/config persistence or an existing Hugging Face-compatible pattern, selecting the smaller fit after repository inspection. Save encoder configuration, head dropout, encoder and head weights, and the pretrained model identifier. The eventual inference loader must restore the trained head rather than initialize a fresh one. Record tokenizer identity and dependency versions. Checkpoint loading errors must be explicit.

## 7. Verification

Use one small runnable test file with an actual tiny ModernBERT configuration, avoiding a large checkpoint download for routine tests. Check forward shape and finite logits, candidate-group isolation, probabilities summing to one, zero and negative logits remaining eligible, explicit invalid masking, rejection of all-invalid states and malformed targets, and the analytic two-candidate example.

Backpropagate the grouped loss and verify finite, nonzero gradients reach both the head and encoder. Verify evaluation-mode checkpoint round-trip predictions agree. Use temporary directories for checkpoint tests. Run these checks in a suitable isolated Python environment and report actual versions and results. If a full pretrained smoke test is not run, state that explicitly rather than implying the offline test verifies downloaded weights.

## 8. Deferred: inference

Load the saved model and matching tokenizer, call eval() and inference_mode(), batch all candidate prompts, preserve candidate ID ordering, apply the validity mask, compute within-state probabilities, and return the argmax candidate with scores and probabilities. Add an API or CLI only when implementing this phase. Handle no-legal-move states at the game boundary.

## 9. Deferred: arena baseline

Integrate with the existing player interface after inspecting its timing and response contract. The untuned baseline consists of a pretrained encoder plus a randomly initialized head: it is not a trained Tetris policy or calibrated preference model. Fix and save its random seed and checkpoint for reproducibility. Compare with the existing heuristic and teacher using shared piece sequences; measure move quality separately from latency/deadline losses. Do not run paid teacher matches in this phase.

## Acceptance criteria

The plan is in the confirmed repository's plans/ before implementation. The model produces one scalar per candidate, supports full encoder/head gradient flow, correctly computes grouped masked distillation loss, and saves/restores all learned weights. Runnable checks pass. README states what exists, what remains deferred, and how the untuned baseline should be interpreted.

## References

- ModernBERT model and configuration API: https://huggingface.co/docs/transformers/model_doc/modernbert
- PyTorch stable log-softmax: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html

## Inspection finding: shortlist labels

`createJevPlayer({betterPrompt: true})` sends only `offered` shortlist candidates
to the teacher but passes all `placements` to `stateRows`, which assigns zero to
missing probabilities. Sum-to-one validation cannot distinguish unoffered moves
from teacher-rated zero-probability moves. The future data pipeline must recover
or record the offered set, and group loss must normalize only over that set.
This phase provides an explicit eligibility mask but does not change the logger.

## Implementation result

Completed: `models/modernbert.py`, `requirements-model.txt`,
`test/test_modernbert.py`, README usage notes, and Python/checkpoint ignore rules.
Offline checks passed on Python 3.12.13, PyTorch 2.14.1, Transformers 4.57.6:
forward shape, gradient flow into encoder/head, grouped probability/loss math,
masking, zero/negative scores, malformed inputs, checkpoint round-trip, and
pretrained construction from a tiny locally saved encoder. `git diff --check`
passed. Full ModernBERT-base weights were not downloaded or smoke-tested, and
no fine-tuning, inference service, or arena integration was performed.

## Phase 2: batched inference (completed)

Added `models/inference.py`: `run_inference(model, tokenizer, prompts,
candidate_ids=None, valid_mask=None)` accepts all candidate prompts for one
current state. It validates state consistency, tokenizes without truncation,
pads once, scores the whole batch in eval/inference mode, computes one softmax,
and returns ordered logits/probabilities plus the chosen index, ID and prompt.
The chosen prompt preserves the supplied candidate's next board. No game-state
mutation or board generation occurs.

The CLI supports saved scorer checkpoints and an explicit seeded untuned
baseline, accepts JSON prompt arrays or dataset rows, and requires state
selection for multi-state datasets. It uses the saved tokenizer identity and
supports an explicit override. Teacher labels are not read as model inputs.

`test/test_inference.py` adds an offline tokenizer + real tiny encoder check,
including single-batch execution, no gradients, eval mode, probability arithmetic,
selection mapping, repeated-call determinism, padding and permutation consistency,
masking, invalid inputs, zero-logit ties, checkpoint CLI, and mixed-state rejection.
Both Python checks passed (2 tests, 13.570 seconds). Full pretrained
ModernBERT-base inference also passed on the actual recorded state `...:2`:
seed 7, CPU, p1=0.5058186650 and p6=0.4941813648, chosen p1. This uses a random
scoring head, not a fine-tuned policy. Arena integration remains deferred.
