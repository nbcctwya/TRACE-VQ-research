"""Stage2-only mechanism experiments with matching capacity and RNG controls."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from trace_vq.models.next_screen import NEXT_VARIANTS, NextPredictor, standardized_turnover


MECHANISM_VARIANTS = {
    "baseline": {},
    "history_gru": copy.deepcopy(NEXT_VARIANTS["history_gru"]),
    "turnover_control": copy.deepcopy(NEXT_VARIANTS["turnover_control"]),
    "history_continuous": dict(copy.deepcopy(NEXT_VARIANTS["history_gru"]), history_feature="continuous"),
    "history_mismatched": dict(copy.deepcopy(NEXT_VARIANTS["history_gru"]), history_mismatch=True),
    "beta_smooth": dict(copy.deepcopy(NEXT_VARIANTS["turnover_control"]),
                        beta_smooth_weight=.05, beta_rms_epsilon=1e-6),
    "peer_self": {"peer": "self"},
    "peer_market": {"peer": "market"},
    "peer_code": {"peer": "code"},
    "state_balanced": {"state_balanced": True},
}

PHASES = [
    ("01_history_mechanisms", ["history_continuous", "history_mismatched"]),
    ("02_loading_stability", ["beta_smooth"]),
    ("03_code_relations", ["peer_self", "peer_market", "peer_code"]),
    ("04_state_weighting", ["state_balanced"]),
]
REFERENCES = ("baseline", "history_gru", "turnover_control")
PRIMARY = {"baseline": "baseline", "history_gru": "baseline", "turnover_control": "baseline",
           "history_continuous": "history_gru", "history_mismatched": "history_gru",
           "beta_smooth": "turnover_control", "peer_self": "baseline", "peer_market": "peer_self",
           "peer_code": "peer_market", "state_balanced": "baseline"}


def loading_stability(current, previous, epsilon=1e-6):
    def normalize(value):
        return value / (value.square().mean(-1, keepdim=True) + epsilon).sqrt()
    return .5 * sum(F.smooth_l1_loss(normalize(a), normalize(b))
                    for a, b in zip(current, previous))


def peer_average(values, codes, kind):
    count = len(values)
    if kind == "self":
        mask = torch.eye(count, device=values.device, dtype=torch.bool)
    elif kind == "market":
        mask = ~torch.eye(count, device=values.device, dtype=torch.bool)
    elif kind == "code":
        mask = (codes[:, None] == codes[None, :]) & ~torch.eye(count, device=values.device, dtype=torch.bool)
    else:
        raise ValueError("Unknown peer relation")
    neighbors = mask.sum(-1, keepdim=True)
    result = (mask.to(values.dtype) @ values) / neighbors.clamp_min(1)
    return result, neighbors.squeeze(-1) > 0


class MechanismPredictor(NextPredictor):
    def __init__(self, config, T_max):
        super().__init__(copy.deepcopy(config), T_max)
        state = torch.get_rng_state()
        try:
            if self.research.get("peer"):
                self.peer_norm = nn.LayerNorm(self.vq_embed_dim)
                self.peer_head = nn.Linear(self.vq_embed_dim, self.vq_embed_dim)
                nn.init.zeros_(self.peer_head.weight)
                nn.init.zeros_(self.peer_head.bias)
        finally:
            torch.set_rng_state(state)
        self._state_weight = None
        self._peer_coverage = None

    def build_stage2_latent(self, h, zq):
        latent = super().build_stage2_latent(h, zq)
        kind = self.research.get("peer")
        if not kind:
            return latent
        # Recompute exact native IDs: STE output rounding can make equal-code
        # z_q vectors differ in their final bits. All three peer arms execute
        # this same deterministic frozen-eval lookup, without consuming RNG.
        with torch.no_grad():
            codes = self.quantizer(h.detach())[2][2]
        values, available = peer_average(self.peer_norm(h.detach()), codes, kind)
        self._peer_coverage = available.float().mean().detach()
        return latent + self.peer_head(values) * available[:, None]

    def batch_prediction(self, batch):
        self._state_weight = batch.get("state_weight") if isinstance(batch, dict) else None
        return super().batch_prediction(batch)

    def training_loss(self, parts, result):
        loss, terms = super().training_loss(parts, result)
        if self.research.get("state_balanced"):
            if self._state_weight is None or self._state_weight.numel() != 1:
                raise ValueError("A single causal training-date weight is required")
            weight = self._state_weight.detach()
            if not torch.isfinite(weight) or not .5 - 1e-6 <= float(weight) <= 2 + 1e-6:
                raise ValueError("Invalid training state weight")
            loss = loss + (weight - 1) * terms["mse"]
            terms["state_weight"] = weight
        if self.research.get("peer"):
            terms["peer_coverage"] = self._peer_coverage
        return loss, terms

    def training_step(self, batch, batch_idx):
        weight = self.research.get("beta_smooth_weight", 0)
        if not weight:
            return super().training_step(batch, batch_idx)
        parts, result = self.batch_prediction(batch)
        loss, terms = self.training_loss(parts, result)
        if batch["align_current"].numel() > 1:
            previous = {name: batch["prev_" + name] for name in ("canonical", "frozen_h", "frozen_zq")}
            old = self.batch_prediction(previous)[1]
            current_index, previous_index = batch["align_current"].long(), batch["align_previous"].long()
            # Preserve the exact old zero-penalty paired-forward control.
            score_loss = standardized_turnover(result[0][current_index], old[0][previous_index])
            loss = loss + self.research.get("turnover_weight", 0) * score_loss
            penalty = loading_stability([result[1][current_index], result[2][current_index]],
                                        [old[1][previous_index], old[2][previous_index]],
                                        self.research["beta_rms_epsilon"])
            terms.update(turnover=score_loss, beta_smooth=penalty)
            loss = loss + weight * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loading-smoothing loss")
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(result[0]))
        self.log_dict({"train_" + name + "_loss": value for name, value in terms.items()},
                      on_step=False, on_epoch=True, batch_size=len(result[0]))
        return loss
