"""Shared STATE/CANDIDATE scorer and per-state distillation objective."""

from pathlib import Path

import torch
from torch import nn
import transformers
from transformers import ModernBertConfig, ModernBertModel


class ModernBertScorer(nn.Module):
    """Right-padded token rows [N, L] -> unrestricted candidate logits [N]."""

    def __init__(self, encoder: ModernBertModel, dropout=0.1, tokenizer_id=None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer_id = tokenizer_id
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, 1, bias=False)

    @classmethod
    def from_pretrained(cls, model_id="answerdotai/ModernBERT-base", dropout=0.1):
        """Load pretrained encoder weights and initialize a NEW, untrained head."""
        encoder = ModernBertModel.from_pretrained(
            model_id, attn_implementation="sdpa", reference_compile=False
        )
        return cls(encoder, dropout, str(model_id))

    def forward(self, input_ids, attention_mask):
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must be nonempty [candidates, tokens]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")
        if attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device:
            raise ValueError("attention_mask must match input_ids shape and device")
        config = self.encoder.config
        if input_ids.shape[1] > config.max_position_embeddings:
            raise ValueError("Input exceeds encoder context; do not silently truncate prompts")
        if not ((input_ids >= 0) & (input_ids < config.vocab_size)).all():
            raise ValueError("Token ID outside encoder vocabulary")
        if not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("attention_mask must be binary")
        mask = attention_mask.bool()
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("Inputs must be right-padded with an attended first token")
        if config.cls_token_id is None or not (input_ids[:, 0] == config.cls_token_id).all():
            raise ValueError("Each prompt must begin with the tokenizer's CLS token")
        return self._score(input_ids, attention_mask)

    def _score(self, input_ids, attention_mask):
        """Pure encoder+head compute, kept free of guards so torch.compile can fuse it."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              return_dict=True).last_hidden_state
        return self.head(self.dropout(hidden[:, 0])).squeeze(-1)

    def compile_scorer(self):
        """Opt-in: fuse the encoder+head compute. Pays a one-time compile cost that only
        amortises over many forward calls; leave off for one-shot CLI runs. dynamic=True
        avoids a recompile per padded sequence length. forward stays eager, so validation
        and forward pre-hooks are unaffected."""
        self._score = torch.compile(self._score, fullgraph=True, dynamic=True)
        return self

    def save_checkpoint(self, path):
        """Save configuration and ALL weights; parent directory must exist."""
        torch.save({
            "format_version": 1,
            "encoder_config": self.encoder.config.to_dict(),
            "dropout": self.dropout.p,
            "tokenizer_id": self.tokenizer_id,
            "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__},
            "state_dict": self.state_dict(),
        }, Path(path))

    @classmethod
    def load_checkpoint(cls, path):
        """Restore on CPU without downloading weights; caller chooses device/mode."""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["format_version"] != 1:
            raise ValueError("Unsupported checkpoint format")
        config = ModernBertConfig.from_dict(saved["encoder_config"])
        config.reference_compile = False
        config._attn_implementation = "sdpa"
        model = cls(ModernBertModel(config), saved["dropout"], saved["tokenizer_id"])
        model.load_state_dict(saved["state_dict"], strict=True)
        return model


def _groups(logits, group_sizes, valid_mask):
    if logits.ndim != 1 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be a finite floating-point vector")
    sizes = tuple(group_sizes)
    if not sizes or any(type(n) is not int or n <= 0 for n in sizes) or sum(sizes) != logits.numel():
        raise ValueError("Positive integer group_sizes must sum to the number of logits")
    if valid_mask is None:
        valid_mask = torch.ones_like(logits, dtype=torch.bool)
    if valid_mask.dtype != torch.bool or valid_mask.shape != logits.shape or valid_mask.device != logits.device:
        raise ValueError("valid_mask must be Boolean and match logits shape and device")
    masks = valid_mask.split(sizes)
    if any(not mask.any() for mask in masks):
        raise ValueError("Every state must have at least one eligible candidate")
    return sizes, masks


def candidate_probabilities(logits, group_sizes, valid_mask=None):
    """Return flat probabilities, separately normalized within each state.

    Mask False means excluded from this choice set, not a zero model score.
    """
    sizes, masks = _groups(logits, group_sizes, valid_mask)
    return torch.cat([
        scores.float().masked_fill(~mask, -torch.inf).softmax(dim=0)
        for scores, mask in zip(logits.split(sizes), masks)
    ])


def distillation_loss(logits, targets, group_sizes, valid_mask=None):
    """Soft-target cross-entropy, averaged equally over states.

    Allow and normalize rounding error of at most 1e-4 per target distribution.
    Masked candidates must have zero target mass. Valid zero targets are retained.
    """
    sizes, masks = _groups(logits, group_sizes, valid_mask)
    if targets.shape != logits.shape or targets.device != logits.device:
        raise ValueError("targets must match logits shape and device")
    if not targets.is_floating_point() or not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("targets must be finite nonnegative floating-point probabilities")
    losses = []
    for scores, q, mask in zip(logits.split(sizes), targets.split(sizes), masks):
        if (q[~mask] != 0).any():
            raise ValueError("Excluded candidates cannot have teacher probability mass")
        q = q[mask].float()
        total = q.sum()
        if not torch.isclose(total, total.new_tensor(1.0), atol=1e-4, rtol=0):
            raise ValueError("Teacher probabilities must sum to one for each state")
        losses.append(-((q / total) * scores[mask].float().log_softmax(dim=0)).sum())
    return torch.stack(losses).mean()
