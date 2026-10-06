# Training methodology

How the ModernBERT Tetris scorer is trained. The authoritative source is
`model_training/model_training.ipynb`; this document explains *why* each piece
is there. Data generation lives in `tools/gen-heuristic-difficulty.mjs` and
`data/collapse_battles.py`.

---

## 1. The model

**Base:** `answerdotai/ModernBERT-base`, full fine-tuning (encoder + head, no
LoRA/adapters). Loaded through Unsloth's `FastModel` with `full_finetuning=True`.

**Head:** a single bias-free `Linear(hidden_size → 1)` on the CLS token
(`hidden[:, 0]`) with dropout `0.1`. It emits **one scalar score per
candidate**. There is no classification layer — scoring is relative within a
state.

**Task framing.** Each training example is one board *state* with ~10 candidate
placements. The state's prompt prefix (`STATE\n…`) plus each candidate's
`CANDIDATE\n…` section are tokenized separately, so every candidate is a full
sequence sharing the same state prefix. The model scores each candidate; a
`log_softmax` over the candidates of one state turns the scores into a
distribution. This is a **listwise learning-to-rank / distillation** setup, not
per-example classification.

**Loss.** Soft cross-entropy (distillation):
`loss = -(target * logits.log_softmax(0)).sum()` over the candidates of a single
state. The target is the teacher's probability distribution over that state's
candidates (renormalized to sum to 1). **Accuracy** = whether the model's
top-scored candidate matches the teacher's top-choice (argmax agreement).

**Checkpoint format.** `model.pt` stores `format_version`, encoder config,
dropout, tokenizer id, library versions, and the state dict — the existing
inference format. Saved atomically (`.tmp` → `replace`).

---

## 2. Two teachers, two rounds (curriculum SFT)

Training is **two sequential rounds**, each a full fine-tune, Round 2 resuming
Round 1's weights:

| Round | Teacher | Data | Starts from |
|---|---|---|---|
| 1 | DT heuristic | `data/heuristic-training/*.json` | fresh pretrained ModernBERT + random head |
| 2 | Jev (System One) | `data/battles-training.jsonl` | Round 1's completed `model.pt` |

Rationale: the heuristic produces cheap, abundant, difficulty-balanced labels
to teach the model Tetris board sense first; Jev's logs are scarcer and are the
real distillation target, applied second so the model specializes last. Each
round gets a fresh optimizer and LR schedule.

### The heuristic teacher (Round 1 labels)

Dellacherie-Thiery afterstate evaluation (`public/tetris.js`,
`FEATURE_WEIGHTS`): nine linear features (six classic Dellacherie + three
BCTS/DT extensions — landing height, eroded cells, row/column transitions,
holes, well sum, hole depth, rows-with-holes, pattern diversity). The driver
uses **2-ply, next-piece-aware lookahead** (`lookaheadScore`) and a softmax over
placement values with a `temperature` knob (raw scores are near one-hot, so
temperature spreads them into a usable soft target).

### The Jev teacher (Round 2 labels)

Jev's actual move distributions, logged from real games and exported by
`collapse_battles.py`. Eligibility is strict (see §4).

---

## 3. Difficulty staging (5 stages)

Both datasets are split into **5 difficulty stages**, but by different means.

**Heuristic — stack-height buckets** (`gen-heuristic-difficulty.mjs`). Difficulty
is the board's `describeHeight` band *before* the move, with fixed quotas:

| Stage | Bucket | Height band | Quota | Garbage seed |
|---|---|---|---|---|
| 1 | very-easy | ≤4 | 2500 | none (natural play from empty) |
| 2 | easy-medium | 5–8 | 2500 | 6 rows @ 0.50 density |
| 3 | medium | 9–12 | 3000 | 10 rows @ 0.65 |
| 4 | hard | 13–15 | 2500 | 14 rows @ 0.80 |
| 5 | near-topout | 16–19 | 2000 | 16 rows @ 0.85 |

The DT heuristic keeps the board low, so hard bands never occur naturally. Those
buckets **pre-seed the board with random garbage** (no complete rows, so nothing
clears and the height is exact), then let the heuristic play and dig out from
there. `near-topout` caps the seed at 16 rows so the spawn row stays clear and
the game doesn't instantly top out.

**Jev — structural-difficulty quantiles** (`collapse_battles.py`). A scalar
`structural_difficulty(state)` is computed from the best candidate's prompt
fields — a weighted sum of stack height, holes, surface roughness, and a
"forced badness" term (holes created + stack growth + resulting height), weights
in `W`. States are cut into 5 stages by **quantiles** of that score across the
corpus. Each state also records `A`, an entropy-based ambiguity measure.

---

## 4. Data preparation and eligibility (before any training)

All preparation happens **once, before Round 1**, so holdouts are shared and
leakage is impossible.

**Episodes.** Every record must carry an `episode_id` (or a trustworthy per-game
`run` fallback). One episode = one game. Legacy heuristic buckets with only
bucket-level run IDs are rejected — they can't supply independent episodes
within a stage, which is why `gen-heuristic-difficulty.mjs` caps
`maxPerGame` (default 10): a low cap means many games → many episodes → a
trustworthy per-game holdout.

**Jev eligibility (strict, never inferred):**
- Every candidate must carry an explicit boolean `offered_to_jev`.
- An unoffered candidate carrying nonzero teacher mass is a hard error.
- Only offered candidates are kept.
- Each kept candidate must have `probability_observed == True`, or the known
  legacy exception (`eligibility_source == "legacy_solo_offered_only"` with no
  `probability_observed` field). Anything else fails.
- Unknown/missing labels are **never** inferred from positive probability.

**Probability mass.** Targets must be finite, in `[0, 1]`, sum > 0, and sum
within tolerance of 1 (`1e-4` heuristic; `~0.010001` for Jev, to accept rounded
0.99/1.01 but reject larger errors). Within-tolerance rounding is renormalized
and *reported* in the manifest; larger discrepancies fail.

**Prompt structure.** Every candidate prompt must start with `STATE\n`, contain
exactly one `\n\nCANDIDATE\n` separator, and share an identical STATE prefix
with the other candidates of its state. Prompts are tokenized once here (CPU)
and must fit `max_seq_length` (2048) or training fails loudly.

---

## 5. The split (episode-level, 81/9/10)

Splitting is **by whole episode, jointly across both datasets**, and never looks
at teacher probabilities.

1. Collect all episode IDs across heuristic + Jev, shuffle with a seeded RNG.
2. `test_fraction = 0.10` of episodes → test.
3. `validation_fraction = 0.20` of the *remaining* pool → validation.
4. The rest → train. Net ≈ **81 / 9 / 10** train/val/test by episode count.

Because the split is by episode, per-stage state counts and proportions vary —
exact quotas are **not** preserved, and that's accepted.

**Exact-text dedup across splits.** If the same state text appears in multiple
splits, it's kept only in its highest-priority split (**test > validation >
train**) and removed from the others. Holdouts never leak into either training
round.

**Singletons.** Single-candidate states (no real decision) are *excluded from
training* but kept in evaluation. Primary metrics cover only **decisions**
(states with ≥2 candidates).

**Hard failure, no fallback.** If any stage lacks train/val/test decisions after
the episode split, training raises — it does **not** silently fall back to
state-level splitting. The manifest (`split.json`) records the full episode
assignment, removed overlaps, excluded singletons, and normalization records.

---

## 6. Blended difficulty curriculum (the key training technique)

Not easiest-first stage blocks. Epoch 1 is one pass that visits **every training
state exactly once**, but reordered so difficulty shifts gradually easy→hard.

The epoch-1 pass is divided into four **time phases** by
`curriculum_phase_fractions = [0.10, 0.30, 0.30, 0.30]`. Each phase has a target
**mix over the 5 stages** (`curriculum_phase_mix`, rows=phase, cols=stage):

```
phase 1 (first 10%):  [0.35, 0.35, 0.25, 0.03, 0.02]   mostly easy
phase 2 (next 30%):   [0.20, 0.20, 0.40, 0.11, 0.09]
phase 3 (next 30%):   [0.10, 0.10, 0.40, 0.22, 0.18]
phase 4 (last 30%):   [0.05, 0.05, 0.25, 0.35, 0.30]   mostly hard
```

**How it's realized (`make_plan`, "reorder-once"):** each stage's states are
shuffled, then spread across the four phases in proportion to
`mix[phase][stage] * phase_fraction` using **Hamilton's largest-remainder**
apportionment. So every state lands in exactly one phase (visit-once preserved),
while the realized per-phase mix tracks the targets as closely as the **fixed
per-stage supply allows** (clamped — you can't draw more hard states than exist).
Within each phase, tiers are shuffled together so difficulties interleave rather
than block.

The effect: easy states thin out but persist late; hard states appear rarely
from the start and dominate the end. The same blended curriculum runs in both
rounds (heuristic states, then Jev states).

**Extra epochs.** `train_epochs = 1` means the curriculum pass only. With `≥2`,
later epochs are plain shuffled full-data passes (no curriculum) — these are
true repeated epochs, distinct from the curriculum phases.

---

## 7. Batch warmup (growing effective batch size)

The effective batch size **ramps** from `start_batch_size = 16` to
`max_batch_size = 32` states over the first `batch_warmup_fraction = 0.10` of
training states, linearly in states seen:

```
size = round(16 + progress * (32 - 16)),  progress = min(1, seen / warmup_states)
```

Small batches early give noisier, faster-moving updates when the model is far
from a solution; larger batches later give stabler gradients near convergence.
After warmup, `seen` is saturated, so any extra shuffled epochs run at the full
32. Batch size is logged per step.

**Microbatching.** One GPU microbatch = **all candidates of one state** (dynamic
right-padding within that state only). Gradient accumulation combines 16–32
states per optimizer update, each state's loss scaled by `1/len(states)`.

---

## 8. Learning-rate schedule (WSD: warmup–stable–decay)

`get_wsd_schedule` over the planned number of optimizer updates:

- **Warmup** `lr_warmup_fraction = 0.10` — linear ramp to peak.
- **Stable** remaining ~70% — held at peak LR (`2e-5`).
- **Decay** `lr_decay_fraction = 0.20` — cosine decay to zero.

Peak LR `2e-5`, `weight_decay = 0.01`.

**Only successful optimizer updates advance the schedule.** An FP16 overflow
skips the update, reduces the loss scale, and does *not* step the scheduler — so
overflow can leave the schedule short of its planned end rather than corrupting
it. Skipped states are not replayed. Counts and loss scale are logged.

---

## 9. Optimizer and numerical profile (T4)

**Optimizer:** `optimi.StableAdamW` (`torch-optimi==0.3.3`, pinned for the
`foreach` API), `betas=(0.9, 0.99)`, `eps=1e-6`, `foreach=True`, Triton backend
**disabled**. StableAdamW does internal update clipping, so **no global gradient
clipping** is applied. `foreach` can raise optimizer peak memory — set
`optimizer_foreach=False` to reduce it.

**Precision (T4 profile):**
- FP32 master weights.
- Forward passes under **FP16 autocast** with **gradient scaling**
  (`GradScaler`).
- Loss computed in FP32.
- **SDPA** attention (automatic kernel selection).
- **TF32 off**, `float32_matmul_precision = "highest"`.

**Memory:** gradient checkpointing (`use_reentrant=False`), one state per
microbatch, pinned memory, non-blocking transfers, 2 CPU dataloader workers with
bounded prefetch (`prefetch_factor=2`). Evaluation uses 0 workers (small
in-memory sets).

Not enabled for T4: BF16, standard FlashAttention 2, padding-free ModernBERT,
cross-state length bucketing. The detected GPU and chosen settings are printed
at setup and saved in each round's `config.json`.

---

## 10. Evaluation, logging, and comparison

**Validation during training.** Live validation samples up to
`live_val_per_stage = 25` decision states per stage; full validation runs at
every phase/stage boundary. Metrics refresh every `log_every = 20` optimizer
updates plus each boundary.

**Metrics.** Primary = **decision loss + top-choice accuracy** on states with ≥2
candidates. All-state loss/accuracy and singleton counts are also retained.
Accuracy = teacher top-choice agreement, **not gameplay win rate**.

**TensorBoard.** Per-attempt timestamped log dir (no mixed curves). Scalars:
train/val-subset/val-full loss & accuracy (combined "Training curves" custom
layout), LR, batch size, curriculum phase/epoch, successful/skipped updates,
loss scale.

**Baseline vs final.** Each round saves `baseline.pt` (starting weights) and
compares baseline vs final `model.pt` on the **identical held-out test
decisions**, per stage and overall, written to `test_comparison.json` and a
TensorBoard table. Final weights are restored afterward.

---

## 11. Reproducibility and run hygiene

- Single seed `3407` seeds Python/torch/CUDA RNG and all split/shuffle/curriculum
  RNGs (offset per use, e.g. `seed+100+stage`).
- `RUN_ID` (`attempt-001`) names the attempt folder. **Existing attempt folders
  are never overwritten** — a fresh attempt needs a new `RUN_ID`.
- `STARTED` reserves an attempt before any expensive write; `COMPLETE` is written
  only after training *and* test comparison succeed. Checkpoints are inference
  weights only — **no optimizer resume**; an interrupted run restarts fresh under
  a new `RUN_ID`.
- Config is validated (`validate_config`) at setup and before each round: integer
  bounds, fraction ranges, warmup+decay ≤ 1, curriculum fractions in (0,1)
  summing to 1, mix rows nonnegative summing to 1, one mix row per phase.

**Per round, saved to Drive:** `baseline.pt`, `model.pt`, `test_comparison.json`,
tokenizer files, `split.json`, `config.json`, `updates.json`, `history.json`,
TensorBoard logs.
