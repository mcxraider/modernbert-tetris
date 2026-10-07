"""ModernBERT: two-round curriculum SFT.

Full fine-tuning of the encoder and scalar candidate-scoring head.

Round 1 visits each heuristic training state once in a blended difficulty curriculum: the
epoch-1 pass is split into time phases (`curriculum_phase_fractions`) whose difficulty mix
shifts from mostly-easy to mostly-hard (`curriculum_phase_mix`), rather than disjoint
easiest-first stage blocks. Easy states thin out but persist late; hard states appear rarely
from the start. Round 2 reloads the completed heuristic checkpoint and applies the same
blended curriculum to the Jev states. Every state is still visited exactly once per epoch;
with `train_epochs>1`, later epochs add shuffled full-data passes. These are curriculum
phases, not repeated full-dataset epochs.

Both datasets are prepared before training. Whole episodes are assigned jointly to
approximately 81/9/10 train/validation/test by episode count. State counts and stage
proportions will vary. Exact state-text overlap is removed from lower-priority splits
(test > validation > train), across both rounds. Single-candidate states remain in
evaluation but are excluded from training; primary metrics cover actual decisions (at least
two candidates).

Needs a CUDA GPU. Run as a script; it executes both rounds in sequence. Fresh attempts
require a new RUN_ID; existing attempt folders are never overwritten.

Install deps first:
    pip install unsloth "transformers>=4.48,<5" tensorboard "torch-optimi==0.3.3"

Prepare data from the repository root before running:
    node tools/gen-heuristic-difficulty.mjs --out-dir data/heuristic-training
    python3 data/collapse_battles.py --output data/battles-training.jsonl

Watch training curves with:
    tensorboard --logdir model_training/tensorboard
"""
from pathlib import Path

# Repo-relative paths (script lives in <repo>/model_training/).
PROJECT_DIR = Path(__file__).resolve().parent.parent
HEURISTIC_DIR = PROJECT_DIR / "data/heuristic-training"
JEV_FILES = [PROJECT_DIR / "data/battles-training.jsonl"]
CHECKPOINT_DIR = PROJECT_DIR / "model_training/checkpoints"
LOG_DIR = PROJECT_DIR / "model_training/tensorboard"

CFG = dict(
    model_id="answerdotai/ModernBERT-base",
    seed=3407,
    test_fraction=0.10,
    validation_fraction=0.20,  # fraction of the remaining train+validation pool
    learning_rate=2e-5,
    weight_decay=0.01,
    optimizer_foreach=True,  # set False to reduce optimizer peak memory if needed
    head_dropout=0.1,
    max_seq_length=2048,
    num_workers=2,  # CPU-only training workers; set 0 for notebook worker issues
    prefetch_factor=2,  # whole states per worker, not optimizer batches
    start_batch_size=16,
    max_batch_size=32,
    batch_warmup_fraction=0.10,
    lr_warmup_fraction=0.10,
    lr_decay_fraction=0.20,  # remaining 70% stays at the peak LR
    log_every=20,  # optimizer updates between live validation + chart refreshes
    live_val_per_stage=25,  # fixed subset; full validation at each stage boundary
    train_epochs=1,  # 1 = curriculum pass only; 2+ adds shuffled pass(es) over all stages
    # Epoch-1 blended curriculum: four time phases, each a mix over the difficulty
    # stages. Rows = phases (spans in curriculum_phase_fractions), columns = stages
    # 1..5. Reorder-once: every training state is still visited exactly once; only
    # the per-phase difficulty mix is biased toward these targets, clamped to the
    # fixed per-stage supply, so the realized mix tracks the targets as closely as
    # the data allows while easy share falls and hard share rises across phases.
    curriculum_phase_fractions=[0.10, 0.30, 0.30, 0.30],  # phase spans of the epoch-1 pass
    curriculum_phase_mix=[                                 # rows=phase, cols=stage 1..5
        [0.35, 0.35, 0.25, 0.03, 0.02],
        [0.20, 0.20, 0.40, 0.11, 0.09],
        [0.10, 0.10, 0.40, 0.22, 0.18],
        [0.05, 0.05, 0.25, 0.35, 0.30],
    ],
)
# Explicit identity: change for a fresh attempt. Rerunning does not mint a new run.
RUN_ID = "attempt-001"
# Only used after candidate eligibility has been established. Logs retain original mass.
CFG["jev_probability_tolerance"] = 0.010001  # accepts rounded 0.99 / 1.01, rejects larger errors


def validate_config():
    for key in ("start_batch_size", "max_batch_size", "prefetch_factor", "log_every",
                "live_val_per_stage", "max_seq_length", "train_epochs"):
        if type(CFG[key]) is not int or CFG[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(CFG["num_workers"]) is not int or CFG["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if CFG["start_batch_size"] > CFG["max_batch_size"]:
        raise ValueError("start_batch_size must not exceed max_batch_size")
    for key in ("test_fraction", "validation_fraction", "batch_warmup_fraction",
                "lr_warmup_fraction", "lr_decay_fraction"):
        if not 0 < CFG[key] < 1:
            raise ValueError(f"{key} must be between zero and one")
    if CFG["lr_warmup_fraction"] + CFG["lr_decay_fraction"] > 1:
        raise ValueError("LR warmup and decay fractions must sum to at most one")
    if not 0 <= CFG["head_dropout"] < 1 or not 0 <= CFG["jev_probability_tolerance"] <= 0.02:
        raise ValueError("Invalid dropout or probability tolerance")
    fracs, mix = CFG["curriculum_phase_fractions"], CFG["curriculum_phase_mix"]
    if not fracs or any(not 0 < f < 1 for f in fracs) or abs(sum(fracs) - 1) > 1e-6:
        raise ValueError("curriculum_phase_fractions must be in (0,1) and sum to one")
    if len(mix) != len(fracs):
        raise ValueError("curriculum_phase_mix must have one row per phase fraction")
    if len({len(row) for row in mix}) != 1:
        raise ValueError("curriculum_phase_mix rows must all have the same width")
    for row in mix:
        if any(w < 0 for w in row) or abs(sum(row) - 1) > 1e-6:
            raise ValueError("each curriculum_phase_mix row must be nonnegative and sum to one")
    if not RUN_ID or Path(RUN_ID).name != RUN_ID or RUN_ID in (".", ".."):
        raise ValueError("RUN_ID must be a single directory name")

validate_config()


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
import os
os.environ["UNSLOTH_DISABLE_FAST_GENERATION"] = "1"
from unsloth import FastModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from optimi import StableAdamW
from transformers import ModernBertModel, AutoTokenizer, get_wsd_schedule
import json, math, random, shutil, hashlib, gc
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timezone

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA GPU is required for training.")
DEVICE = torch.device("cuda")
AMP_DTYPE = torch.float16
# T4 profile: FP32 master weights, FP16 autocast, SDPA, no TF32.
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False
RUNTIME = dict(
    gpu=torch.cuda.get_device_name(),
    compute_capability=list(torch.cuda.get_device_capability()),
    autocast_dtype="float16", parameter_dtype="float32",
    attention="sdpa", tf32=False, optimizer_triton=False,
    optimizer_foreach=CFG["optimizer_foreach"],
)
print("T4 training profile:", json.dumps(RUNTIME))
random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# T4 settings
# ---------------------------------------------------------------------------
# FP16 autocast + gradient scaling, SDPA attention, StableAdamW foreach with its Triton
# backend explicitly disabled. torch-optimi is pinned to 0.3.3 to preserve the foreach
# API. Foreach can increase optimizer peak memory; set optimizer_foreach=False if needed.
# Gradient checkpointing, one whole state per GPU microbatch, 16->32-state accumulation,
# worker prefetching, pinned memory, and nonblocking transfers remain enabled. Padding
# stays dynamic within each state's candidates. BF16, TF32, and FlashAttention 2 are not
# enabled for T4. The selected settings and detected GPU are saved in each round's
# config.json. GPU memory fit and speed still need verification on actual hardware.


# ---------------------------------------------------------------------------
# Shared model
# ---------------------------------------------------------------------------
# Preserves the existing CLS-pooling scorer and model.pt inference format. One complete
# state's candidates form a GPU microbatch; gradient accumulation supplies the effective
# batch size of 16-32 states. The scorer keeps FP32 master weights and computes forward
# passes under FP16 autocast; gradient scaling is enabled in both rounds. Loss stays FP32.
class modernBERTTetris(nn.Module):
    def __init__(self, encoder, dropout=0.1, tokenizer_id=None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer_id = tokenizer_id
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              return_dict=True).last_hidden_state
        return self.head(self.dropout(hidden[:, 0])).squeeze(-1)

    def save_checkpoint(self, path):
        import transformers
        state = dict(
            format_version=1, encoder_config=self.encoder.config.to_dict(),
            dropout=self.dropout.p, tokenizer_id=self.tokenizer_id,
            versions={"torch": str(torch.__version__), "transformers": transformers.__version__},
            state_dict={k: v.detach().cpu() for k, v in self.state_dict().items()},
        )
        temp = Path(path).with_suffix(".tmp")
        torch.save(state, temp)
        temp.replace(path)

# Tokenization is CPU-only; model allocation happens only after data validation.
tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], padding_side="right")


def fresh_model():
    random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])
    encoder, loaded_tokenizer = FastModel.from_pretrained(
        model_name=CFG["model_id"], auto_model=ModernBertModel,
        max_seq_length=CFG["max_seq_length"], dtype=torch.float32,
        load_in_4bit=False, full_finetuning=True,
        attn_implementation="sdpa",
    )
    encoder.config.reference_compile = False  # config field in TF 4.57+, not a from_pretrained constructor kwarg
    if loaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Training and preparation tokenizers differ")
    scorer = modernBERTTetris(encoder, CFG["head_dropout"], CFG["model_id"]).to(DEVICE).float()
    scorer.requires_grad_(True)
    FastModel.for_training(scorer.encoder, use_gradient_checkpointing=True)
    scorer.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return scorer


# ---------------------------------------------------------------------------
# Shared data loading and holdouts
# ---------------------------------------------------------------------------
# All records require an episode_id, or a trustworthy per-game run fallback. JSONL rows and
# grouped JSON records are accepted; Jev records must carry stage 1-5. Both rounds share one
# episode assignment. Repeated state text is retained only in its highest-priority split:
# test, then validation, then training. Jev eligibility must be explicit on every candidate.
def local_copy(path):
    # No Colab Drive staging locally; validate existence and read in place.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}. Edit the paths in the configuration block.")
    return path


def read_states(path, fixed_stage=None):
    local = local_copy(path)
    if local.suffix == ".jsonl":
        records = [json.loads(line) for line in local.open() if line.strip()]
    else:
        payload = json.loads(local.read_text())
        records = ([dict(state_id=sid, candidates=rows) for sid, rows in payload.items()]
                   if isinstance(payload, dict) else payload)
    grouped = {}
    for record in records:
        sid = str(record["state_id"])
        stage = fixed_stage if fixed_stage is not None else record["stage"]
        if isinstance(stage, bool) or str(stage) not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"{sid}: expected stage 1-5, got {stage!r}")
        rows = record.get("candidates", [record])
        if isinstance(rows, dict):
            rows = [dict(row, candidate_id=cid) for cid, row in rows.items()]
        episodes = {r.get("episode_id") or record.get("episode_id") or r.get("run") or record.get("run")
                    for r in rows}
        if len(episodes) != 1 or not all(isinstance(e, str) and e for e in episodes):
            raise ValueError(f"{sid}: missing/inconsistent episode_id; regenerate legacy heuristic buckets")
        episode = next(iter(episodes))
        if "candidates" in record and sid in grouped:
            raise ValueError(f"Duplicate grouped state: {sid}")
        group = grouped.setdefault(sid, dict(state_id=sid, episode_id=episode, stage=int(stage), candidates=[]))
        if group["stage"] != int(stage) or group["episode_id"] != episode:
            raise ValueError(f"{sid}: inconsistent stage/episode across candidates")
        group["candidates"].extend(rows)
    return list(grouped.values())


def prepare_states(groups, jev=False):
    prepared, seen, adjusted = [], set(), []
    for group in groups:
        sid = group["state_id"]
        if sid in seen:
            raise ValueError(f"Duplicate state ID across files: {sid}")
        seen.add(sid)
        rows = group["candidates"]
        ids = [r.get("candidate_id") for r in rows]
        if any(cid is None for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"{sid}: candidate IDs must be present and unique")
        if jev:
            if any(type(r.get("offered_to_jev")) is not bool for r in rows):
                raise ValueError(f"{sid}: missing offered_to_jev flags; re-export from raw logs")
            if any(not r["offered_to_jev"] and float(r["probability"]) != 0 for r in rows):
                raise ValueError(f"{sid}: unoffered candidate carries teacher mass")
            rows = [r for r in rows if r["offered_to_jev"]]
            if any(r.get("probability_observed") is not True and
                   not ("probability_observed" not in r and r.get("eligibility_source") == "legacy_solo_offered_only")
                   for r in rows):
                raise ValueError(f"{sid}: incomplete teacher response or unknown label provenance")
        if not rows:
            raise ValueError(f"{sid}: empty candidate set")
        targets = [float(r["probability"]) for r in rows]
        mass = sum(targets)
        tolerance = CFG["jev_probability_tolerance"] if jev else 1e-4
        if any(not math.isfinite(q) or q < 0 or q > 1 for q in targets) or mass <= 0 or abs(mass - 1) > tolerance:
            raise ValueError(f"{sid}: invalid candidate probability mass {mass}")
        if abs(mass - 1) > 1e-4:
            adjusted.append(dict(state_id=sid, original_mass=mass))
        targets = [q / mass for q in targets]
        prompts = [r["prompt"] for r in rows]
        prefixes = {p.split("\n\nCANDIDATE\n", 1)[0] for p in prompts}
        if len(prefixes) != 1 or any(not p.startswith("STATE\n") or "\n\nCANDIDATE\n" not in p for p in prompts):
            raise ValueError(f"{sid}: expected matching STATE prefixes and CANDIDATE sections")
        encoded = tokenizer(prompts, truncation=False, padding=False)
        if max(map(len, encoded["input_ids"])) > CFG["max_seq_length"]:
            raise ValueError(f"{sid}: prompt exceeds max_seq_length; increase it explicitly")
        features = [dict(input_ids=ids, attention_mask=mask) for ids, mask in
                    zip(encoded["input_ids"], encoded["attention_mask"])]
        prepared.append(dict(state_id=sid, episode_id=group["episode_id"], stage=group["stage"],
                             features=features, targets=targets, original_mass=mass,
                             state_text=next(iter(prefixes))))
    print(f"Prepared {len(prepared):,} states; normalized {len(adjusted):,} rounded distributions")
    return prepared


def split_rounds(datasets):
    # Assign complete episodes jointly, without inspecting teacher probabilities.
    episodes = sorted({g["episode_id"] for data in datasets.values() for g in data})
    if len(episodes) < 3:
        raise ValueError("Need at least three independent episodes; regenerate legacy heuristic buckets")
    random.Random(CFG["seed"]).shuffle(episodes)
    n_test = max(1, min(len(episodes)-2, round(len(episodes)*CFG["test_fraction"])))
    n_val = max(1, min(len(episodes)-n_test-1, round((len(episodes)-n_test)*CFG["validation_fraction"])))
    assignment = {e: ("test" if i < n_test else "validation" if i < n_test+n_val else "train")
                  for i, e in enumerate(episodes)}
    priority = {"train": 0, "validation": 1, "test": 2}
    text_owner = {}
    for data in datasets.values():
        for g in data:
            split = assignment[g["episode_id"]]
            text = g["state_text"]
            if text not in text_owner or priority[split] > priority[text_owner[text]]:
                text_owner[text] = split
    rounds, manifests = {}, {}
    for name, data in datasets.items():
        stages = {s: dict(train=[], validation=[], test=[]) for s in sorted({g["stage"] for g in data})}
        removed_overlap, singletons = [], []
        for g in data:
            split = assignment[g["episode_id"]]
            if text_owner[g["state_text"]] != split:
                removed_overlap.append(g["state_id"])
            elif split == "train" and len(g["targets"]) == 1:
                singletons.append(g["state_id"])
            else:
                stages[g["stage"]][split].append(g)
        for stage, splits in stages.items():
            for key, rows in splits.items():
                rows.sort(key=lambda g: g["state_id"])
                if not any(len(g["targets"]) > 1 for g in rows):
                    raise ValueError(f"{name} stage {stage}: no {key} decisions after episode split. "
                                     "Collect more independent episodes; regenerate old heuristic buckets with per-game IDs.")
            print(name, stage, {key: len(rows) for key, rows in splits.items()})
        rounds[name] = stages
        manifests[name] = dict(
            episode_assignment=assignment,
            stages={s: {key: [dict(state_id=g["state_id"], episode_id=g["episode_id"]) for g in rows]
                        for key, rows in splits.items()} for s, splits in stages.items()},
            removed_cross_split_text=removed_overlap, excluded_training_singletons=singletons,
            normalized_probabilities=[dict(state_id=g["state_id"], original_mass=g["original_mass"])
                                      for g in data if abs(g["original_mass"]-1) > 1e-4],
        )
        print(f"{name}: excluded {len(singletons)} training singletons and {len(removed_overlap)} cross-split duplicates")
    return rounds, manifests


# ---------------------------------------------------------------------------
# Dataset and DataLoader
# ---------------------------------------------------------------------------
# One dataset item contains all candidates of one state. The loader right-pads only within
# that state and returns CPU tensors. The training loader follows the seeded curriculum plan
# exactly, without an additional shuffle (the per-phase blend and interleave happen inside
# make_plan). Gradient accumulation combines 16-32 states per optimizer update.
class TetrisDataset(Dataset):
    def __init__(self, states, pad_token_id):
        self.states = states
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        state = self.states[index]
        features = state["features"]
        return {
            "state_id": state["state_id"],
            "input_ids": pad_sequence(
                [torch.tensor(row["input_ids"], dtype=torch.long) for row in features],
                batch_first=True, padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [torch.tensor(row["attention_mask"], dtype=torch.long) for row in features],
                batch_first=True, padding_value=0,
            ),
            "targets": torch.tensor(state["targets"], dtype=torch.float32),
        }


def make_loader(states, workers=0):
    options = {"prefetch_factor": CFG["prefetch_factor"]} if workers else {}
    return DataLoader(
        TetrisDataset(states, tokenizer.pad_token_id),
        batch_size=None,  # one entire state; no extra batch dimension
        shuffle=False,
        num_workers=workers,
        pin_memory=DEVICE.type == "cuda",
        generator=torch.Generator().manual_seed(CFG["seed"]),
        **options,
    )


# ---------------------------------------------------------------------------
# Shared training loop
# ---------------------------------------------------------------------------
# Training accumulates 16->32 multi-candidate states per update. The LR schedule warms up
# for 10%, stays stable for 70%, and cosine-decays for 20% of planned updates. An FP16
# overflow skips the optimizer update and reduces the loss scale; only successful updates
# advance LR. StableAdamW uses betas=(0.9, 0.99), eps=1e-6, and internal update clipping;
# no global gradient clipping is applied. Primary validation/test loss and accuracy cover
# states with at least two candidates.
def largest_remainder(weights, n):
    # Nonnegative ints summing to n, apportioned to weights (Hamilton's method).
    total = sum(weights)
    if n <= 0 or total <= 0:
        return [0] * len(weights)
    exact = [w / total * n for w in weights]
    floors = [math.floor(e) for e in exact]
    for i in sorted(range(len(weights)), key=lambda i: exact[i] - floors[i], reverse=True)[:n - sum(floors)]:
        floors[i] += 1
    return floors


def make_plan(stages, shuffled_only=False):
    # Each plan entry is (epoch, group, states). For the epoch-1 blended curriculum
    # group is the time phase (1..len(phase_fractions)); for shuffled passes it is None.
    # shuffled_only skips the epoch-1 curriculum: epochs 1..train_epochs are shuffled passes.
    total = sum(len(s["train"]) for s in stages.values())
    warmup_states = max(1, math.ceil(total * CFG["batch_warmup_fraction"]))
    plan, seen = [], 0

    def batch(epoch, group, items):
        nonlocal seen
        offset = 0
        while offset < len(items):
            progress = min(1.0, seen / warmup_states)
            size = round(CFG["start_batch_size"] + progress * (CFG["max_batch_size"] - CFG["start_batch_size"]))
            chunk = items[offset:offset + size]
            plan.append((epoch, group, chunk))
            offset += len(chunk)
            seen += len(chunk)

    # Epoch-1 blended curriculum. Spread each difficulty stage's states across the
    # phases in proportion to its per-phase target weight (mix[phase][stage] *
    # phase_fraction), so every state is placed in exactly one phase (visit-once)
    # while the easy->hard trend holds. Realized mix is clamped by fixed supply.
    if not shuffled_only:
        stage_keys = sorted(stages)
        fracs, mix = CFG["curriculum_phase_fractions"], CFG["curriculum_phase_mix"]
        if any(len(row) != len(stage_keys) for row in mix):
            raise ValueError(f"curriculum_phase_mix width {len(mix[0])} != {len(stage_keys)} stages present")
        phase_buckets = [[] for _ in fracs]
        for ti, stage in enumerate(stage_keys):
            train = stages[stage]["train"]
            order = list(range(len(train)))
            random.Random(CFG["seed"] + 100 + stage).shuffle(order)
            shuffled = [train[i] for i in order]
            counts = largest_remainder([mix[p][ti] * fracs[p] for p in range(len(fracs))], len(shuffled))
            off = 0
            for p, k in enumerate(counts):
                phase_buckets[p].extend(shuffled[off:off + k])
                off += k
        for p, bucket in enumerate(phase_buckets, start=1):
            random.Random(CFG["seed"] + 300 + p).shuffle(bucket)  # interleave tiers within the phase
            batch(1, p, bucket)
    all_train = [g for split in stages.values() for g in split["train"]]
    start_epoch = 1 if shuffled_only else 2
    for epoch in range(start_epoch, CFG["train_epochs"] + 1):
        shuffled = all_train[:]
        random.Random(CFG["seed"] + 200 + epoch).shuffle(shuffled)
        batch(epoch, None, shuffled)  # seen is saturated, so these run at max_batch_size
    return plan


def state_metrics(group):
    inputs = {key: group[key].to(DEVICE, non_blocking=True)
              for key in ("input_ids", "attention_mask")}
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        logits = model(**inputs).float()
    target = group["targets"].to(DEVICE, non_blocking=True)
    target = target / target.sum()
    loss = -(target * logits.log_softmax(0)).sum()
    accuracy = float(target[logits.detach().argmax()] == target.max())
    return loss, accuracy


@torch.no_grad()
def evaluate(data):
    if not data:
        raise ValueError("Cannot evaluate an empty split")
    model.eval()
    loss = correct = decision_loss = decision_correct = 0.0
    decisions = 0
    for group in make_loader(data, workers=0):
        value, accuracy = state_metrics(group)
        loss += float(value)
        correct += accuracy
        if len(group["targets"]) > 1:
            decisions += 1
            decision_loss += float(value)
            decision_correct += accuracy
    if not decisions:
        raise ValueError("Evaluation requires at least one multi-candidate decision")
    return dict(loss=decision_loss / decisions, accuracy=decision_correct / decisions,
                decisions=decisions, states=len(data), singletons=len(data)-decisions,
                all_state_loss=loss / len(data), all_state_accuracy=correct / len(data))


def amp_optimizer_step(scaler, optimizer, scheduler):
    # GradScaler checks gradients and skips an overflowing update itself.
    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= scale_before
    if updated:
        scheduler.step()
    return updated


def log_metrics(writer, split, metrics, step):
    writer.add_scalar(f"Loss/{split}", metrics["loss"], step)
    writer.add_scalar(f"Accuracy/{split}", metrics["accuracy"], step)
    if "all_state_accuracy" in metrics:
        writer.add_scalar(f"AllStateAccuracy/{split}", metrics["all_state_accuracy"], step)


def compare_test_checkpoints(stages, output, writer, step):
    results = {}
    final_path = output / "model.pt"
    try:
        for label, path in (("baseline", output / "baseline.pt"), ("final", final_path)):
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(saved["state_dict"])
            del saved
            per_stage = {str(stage): evaluate(split["test"])
                         for stage, split in stages.items()}
            total = sum(m["states"] for m in per_stage.values())
            decisions = sum(m["decisions"] for m in per_stage.values())
            overall = {metric: sum(m[metric]*m["decisions"] for m in per_stage.values())/decisions
                       for metric in ("loss", "accuracy")}
            overall.update({metric: sum(m[metric]*m["states"] for m in per_stage.values())/total
                            for metric in ("all_state_loss", "all_state_accuracy")})
            results[label] = dict(overall=dict(states=total, decisions=decisions,
                                               singletons=total-decisions, **overall), per_stage=per_stage)
    finally:
        saved = torch.load(final_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"])
        model.eval()

    table = ["| Test set | Decisions | Baseline loss | Final loss | Loss change | Baseline accuracy | Final accuracy |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for scope in ["overall"] + [str(stage) for stage in stages]:
        before = results["baseline"]["overall"] if scope == "overall" else results["baseline"]["per_stage"][scope]
        after = results["final"]["overall"] if scope == "overall" else results["final"]["per_stage"][scope]
        name = "Overall" if scope == "overall" else f"Stage {scope}"
        table.append(f"| {name} | {before['decisions']} | {before['loss']:.5f} | {after['loss']:.5f} | "
                     f"{after['loss']-before['loss']:+.5f} | {before['accuracy']:.1%} | {after['accuracy']:.1%} |")
        for metric in ("loss", "accuracy"):
            writer.add_scalar(f"Test/{scope}/{metric}/baseline", before[metric], step)
            writer.add_scalar(f"Test/{scope}/{metric}/final", after[metric], step)
    results["loss_change"] = results["final"]["overall"]["loss"] - results["baseline"]["overall"]["loss"]
    (output / "test_comparison.json").write_text(json.dumps(results, indent=2))
    markdown = "\n".join(table)
    print(markdown)
    writer.add_text("Test/baseline_vs_final", markdown, step)
    writer.flush()


def train_round(stages, output, title, manifest, initial_checkpoint=None, learning_rate=None, shuffled_only=False):
    global model
    validate_config()
    # Exclusive reservation happens before any split, baseline, or progress write.
    output.mkdir(parents=True, exist_ok=False)
    (output / "STARTED").write_text("Fresh attempt; no optimizer resume. Use a new RUN_ID after interruption.\n")
    (output / "split.json").write_text(json.dumps(manifest, indent=2))
    if "model" in globals():
        del model
        gc.collect()
        torch.cuda.empty_cache()
    model = fresh_model()
    if initial_checkpoint is not None:
        saved = torch.load(initial_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        shutil.copy2(initial_checkpoint, output / "initial-heuristic.pt")
    model.save_checkpoint(output / "baseline.pt")
    plan = make_plan(stages, shuffled_only=shuffled_only)
    ordered_train = [s for _, _, states in plan for s in states]
    train_loader = make_loader(ordered_train, workers=CFG["num_workers"])
    full_val = [g for split in stages.values() for g in split["validation"]]
    live_val = [g for split in stages.values()
                for g in [v for v in split["validation"] if len(v["targets"]) > 1][:CFG["live_val_per_stage"]]]
    optimizer = StableAdamW(model.parameters(), lr=CFG["learning_rate"] if learning_rate is None else learning_rate,
                                 weight_decay=CFG["weight_decay"], betas=(0.9, 0.99), eps=1e-6,
                                 triton=False, foreach=CFG["optimizer_foreach"])
    warmup = max(1, round(len(plan) * CFG["lr_warmup_fraction"]))
    decay = max(1, round(len(plan) * CFG["lr_decay_fraction"]))
    stable = len(plan) - warmup - decay
    if stable < 0:
        raise ValueError("Not enough optimizer updates for warmup/stable/decay")
    scheduler = get_wsd_schedule(optimizer, num_warmup_steps=warmup,
                                num_stable_steps=stable, num_decay_steps=decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = dict(train=[], validation=[], full_validation=[])
    run_dir = LOG_DIR / output.parent.name / output.name / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    print(f"{title} — TensorBoard logs: {run_dir}")
    with SummaryWriter(log_dir=str(run_dir), flush_secs=10) as writer:
        writer.add_custom_scalars({"Training curves": {
            "Loss": ["Multiline", ["Loss/train", "Loss/validation_subset", "Loss/validation_full"]],
            "Accuracy": ["Multiline", ["Accuracy/train", "Accuracy/validation_subset", "Accuracy/validation_full"]],
        }})
        history["validation"].append(dict(step=0, **evaluate(live_val)))
        log_metrics(writer, "validation_subset", history["validation"][-1], 0)
        writer.flush()
        running_loss = running_correct = count = 0
        tokenizer.save_pretrained(output)
        (output / "config.json").write_text(json.dumps(dict(CFG, runtime=RUNTIME, optimizer="optimi.StableAdamW", betas=[0.9, 0.99], eps=1e-6, learning_rate=CFG["learning_rate"] if learning_rate is None else learning_rate), indent=2))
        train_iterator = iter(train_loader)
        successful_updates = skipped_updates = 0
        for step, (epoch, group, states) in enumerate(plan, start=1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr_used = optimizer.param_groups[0]["lr"]
            for _ in states:
                loss, accuracy = state_metrics(next(train_iterator))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss in epoch {epoch} group {group}")
                scaler.scale(loss / len(states)).backward()
                running_loss += float(loss.detach())
                running_correct += accuracy
                count += 1
            if amp_optimizer_step(scaler, optimizer, scheduler):
                successful_updates += 1
            else:
                skipped_updates += 1
                print(f"FP16 overflow at planned update {step}; reduced scale to {scaler.get_scale()}")
            writer.add_scalar("Schedule/successful_updates", successful_updates, step)
            writer.add_scalar("Schedule/skipped_updates", skipped_updates, step)
            writer.add_scalar("Schedule/loss_scale", scaler.get_scale(), step)
            writer.add_scalar("Schedule/learning_rate", lr_used, step)
            writer.add_scalar("Schedule/batch_size", len(states), step)
            writer.add_scalar("Curriculum/phase", group or 0, step)
            writer.add_scalar("Curriculum/epoch", epoch, step)
            group_end = step == len(plan) or plan[step][:2] != (epoch, group)
            if step % CFG["log_every"] == 0 or group_end:
                history["train"].append(dict(step=step, epoch=epoch, phase=group, loss=running_loss/count,
                                             accuracy=running_correct/count, batch_size=len(states), lr=lr_used))
                history["validation"].append(dict(step=step, epoch=epoch, phase=group, **evaluate(live_val)))
                running_loss = running_correct = count = 0
                if group_end:
                    metrics = evaluate(full_val)
                    history["full_validation"].append(dict(step=step, epoch=epoch, phase=group, **metrics))
                    model.save_checkpoint(output / "model.pt")
                    (output / "progress.json").write_text(json.dumps(dict(completed_epoch=epoch, completed_phase=group, step=step)))
                    label = f"Epoch {epoch} phase {group}" if group is not None else f"Epoch {epoch}"
                    print(f"{label} complete — full val loss {metrics['loss']:.4f}, accuracy {metrics['accuracy']:.1%}")
                log_metrics(writer, "train", history["train"][-1], step)
                log_metrics(writer, "validation_subset", history["validation"][-1], step)
                if group_end:
                    log_metrics(writer, "validation_full", history["full_validation"][-1], step)
                writer.flush()
                (output / "history.json").write_text(json.dumps(history, indent=2))
        del train_iterator, train_loader
        optimizer.zero_grad(set_to_none=True)
        del scheduler, optimizer, scaler
        if successful_updates == 0:
            raise FloatingPointError("Every update overflowed; no completed model can be reported")
        (output / "updates.json").write_text(json.dumps(dict(planned=len(plan), successful=successful_updates,
                                                            skipped=skipped_updates), indent=2))
        compare_test_checkpoints(stages, output, writer, successful_updates)
    (output / "COMPLETE").write_text("Finished all curriculum stages.\n")
    print("Saved:", output / "model.pt")
    return history


# ---------------------------------------------------------------------------
# Prepare both rounds before training
# ---------------------------------------------------------------------------
# A shared episode assignment prevents Jev holdouts from leaking into heuristic training.
# Existing attempt folders are rejected before expensive preparation or writes.
validate_config()
ATTEMPT_DIR = CHECKPOINT_DIR / RUN_ID
HEURISTIC_OUTPUT = ATTEMPT_DIR / "modernBERT-base-heuristic"
DISTILL_OUTPUT = ATTEMPT_DIR / "modernBERT-base-distill"
if ATTEMPT_DIR.exists():
    raise FileExistsError(f"{ATTEMPT_DIR} already exists. Choose a new RUN_ID for a fresh attempt.")
heuristic_files = ["very-easy.json", "easy-medium.json", "medium.json", "hard.json", "near-topout.json"]
heuristic_groups = []
for stage, filename in enumerate(heuristic_files, start=1):
    heuristic_groups.extend(read_states(HEURISTIC_DIR / filename, fixed_stage=stage))
jev_groups = []
for path in JEV_FILES:
    jev_groups.extend(read_states(path))
rounds, split_manifests = split_rounds(dict(
    heuristic=prepare_states(heuristic_groups), jev=prepare_states(jev_groups, jev=True)))
heuristic_stages, jev_stages = rounds["heuristic"], rounds["jev"]
del heuristic_groups, jev_groups, rounds


# ---------------------------------------------------------------------------
# Round 1: heuristics
# ---------------------------------------------------------------------------
heuristic_history = train_round(heuristic_stages, HEURISTIC_OUTPUT, "Round 1: heuristics",
                                split_manifests["heuristic"])


# ---------------------------------------------------------------------------
# Standalone shuffled-method pass over the heuristic stages
# ---------------------------------------------------------------------------
# Loads a saved checkpoint and runs one shuffled full-data pass over the heuristic stages
# (set CFG["train_epochs"] > 1 before running for N shuffled epochs). Own output dir, so it
# never collides with Round 1 / Round 2. Checkpoint load is weights-only.
SHUFFLE_CHECKPOINT = HEURISTIC_OUTPUT / "model.pt"   # point at any saved model.pt
SHUFFLE_OUTPUT = ATTEMPT_DIR / "modernBERT-base-shuffled"

# Shuffled-pass overrides: faster warmup + faster decay => longer at the stable LR.
# Applied to CFG only for this call, then restored so Round 1/2 reruns are unaffected.
SHUFFLE_OVERRIDES = dict(start_batch_size=24, lr_warmup_fraction=0.05, lr_decay_fraction=0.10)
_saved = {k: CFG[k] for k in SHUFFLE_OVERRIDES}
CFG.update(SHUFFLE_OVERRIDES)
try:
    shuffle_history = train_round(heuristic_stages, SHUFFLE_OUTPUT, "Shuffled epoch",
                                  split_manifests["heuristic"],
                                  initial_checkpoint=SHUFFLE_CHECKPOINT,
                                  learning_rate=1e-5, shuffled_only=True)
finally:
    CFG.update(_saved)


# ---------------------------------------------------------------------------
# Round 2: Jev distillation
# ---------------------------------------------------------------------------
# Starts from the completed Round 1 checkpoint, with a fresh optimizer and schedules. One
# pass through the Jev training states. The same episode assignment prepared before Round 1
# is retained. initial-heuristic.pt and baseline.pt preserve the starting weights.
print("Jev curriculum:", list(jev_stages))

if not (SHUFFLE_OUTPUT / "COMPLETE").exists():
    raise RuntimeError("Finish the shuffled pass before starting Round 2.")
distill_history = train_round(jev_stages, DISTILL_OUTPUT, "Round 2: Jev distillation",
                              split_manifests["jev"], initial_checkpoint=SHUFFLE_OUTPUT / "model.pt")


# ---------------------------------------------------------------------------
# Standalone shuffled continuation on the DISTILL checkpoint
# ---------------------------------------------------------------------------
# Mirrors the heuristic shuffle pass above, but starts from Round 2's distilled model.pt and
# runs N shuffled full-data passes over the Jev stages. Own output dir, so no collision with
# any round. Checkpoint load is weights-only.
N_SHUFFLE_EPOCHS = 2   # N shuffled epochs over the full Jev data

if not (DISTILL_OUTPUT / "COMPLETE").exists():
    raise RuntimeError("Finish Round 2 (distill) before the shuffled continuation.")

SHUFFLE_CHECKPOINT = DISTILL_OUTPUT / "model.pt"
SHUFFLE_OUTPUT = ATTEMPT_DIR / "modernBERT-base-distill-shuffled"

# Same shuffled-pass LR shaping as the heuristic shuffle pass; set N epochs explicitly.
SHUFFLE_OVERRIDES = dict(train_epochs=N_SHUFFLE_EPOCHS, start_batch_size=24,
                         lr_warmup_fraction=0.05, lr_decay_fraction=0.10)
_saved = {k: CFG[k] for k in SHUFFLE_OVERRIDES}
CFG.update(SHUFFLE_OVERRIDES)
try:
    distill_shuffle_history = train_round(jev_stages, SHUFFLE_OUTPUT, "Distill shuffled epoch",
                                          split_manifests["jev"],
                                          initial_checkpoint=SHUFFLE_CHECKPOINT,
                                          learning_rate=1e-5, shuffled_only=True)
finally:
    CFG.update(_saved)
