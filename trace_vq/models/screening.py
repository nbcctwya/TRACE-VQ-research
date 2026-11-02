"""Opt-in Stage 2 research variants; the historical implementation stays intact.

New branches are appended after the baseline constructor, and residual branches
start at zero. All variants keep the frozen Stage 1 and the five-value prediction
interface. Validation always uses the original target-day MSE plus auxiliary
loss, so checkpoint selection remains comparable across training objectives.
"""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from trace_vq.data.schema import MARKET_DIM, unpack_batch
from trace_vq.training.prediction import GenerateReturn, softcap_log1p


VARIANTS = {
    "baseline": {},
    "market_route": {"market_conditioning": "route"},
    "market_direct": {"market_conditioning": "direct"},
    "ic_loss": {"ic_weight": 0.1},
    "pairwise_loss": {"pairwise_weight": 0.1},
    "multihorizon": {"multihorizon_weight": 0.1, "auxiliary_horizons": [2, 3, 10]},
    "sparse_prior": {"prior_gate": True, "prior_gate_penalty": 0.001},
    "market_route_ic": {"market_conditioning": "route", "ic_weight": 0.1},
    "market_route_multihorizon": {
        "market_conditioning": "route", "multihorizon_weight": 0.1,
        "auxiliary_horizons": [2, 3, 10],
    },
}
INITIAL_VARIANTS = (
    "baseline", "market_route", "market_direct", "ic_loss", "pairwise_loss",
    "multihorizon", "sparse_prior",
)


def variant_config(config, name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown research variant: {name}")
    result = deepcopy(config)
    result["research"] = dict(variant=name, **deepcopy(VARIANTS[name]))
    return result


def masked_mse(prediction, target):
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("Training batch has no finite supervised targets")
    return F.mse_loss(prediction[valid], target[valid])


def ic_objective(prediction, target):
    valid = torch.isfinite(target)
    p, y = prediction[valid], target[valid]
    if p.numel() < 2:
        return prediction.sum() * 0
    p, y = p - p.mean(), y - y.mean()
    return 1 - (p * y).sum() / ((p.square().sum() + 1e-8).sqrt() * (y.square().sum() + 1e-8).sqrt())


def pairwise_objective(prediction, target):
    valid = torch.isfinite(target)
    p, y = prediction[valid], target[valid]
    if p.numel() < 2:
        return prediction.sum() * 0
    return F.relu(-(p[:, None] - p[None, :]) * (y[:, None] - y[None, :])).mean()


class MarketConditionedGate(nn.Module):
    """Original gate plus a zero-initialized market-conditioned residual."""

    def __init__(self, base, num_experts):
        super().__init__()
        self.base = base
        self.market = nn.Sequential(
            nn.Linear(MARKET_DIM, 16), nn.LayerNorm(16), nn.GELU(),
            nn.Linear(16, num_experts),
        )
        nn.init.zeros_(self.market[-1].weight)
        nn.init.zeros_(self.market[-1].bias)
        self.condition = None

    def forward(self, latent):
        if self.condition is None:
            raise ValueError("The market-conditioned gate requires the current market input")
        return self.base(latent) + self.market(self.condition)


class ScreeningPredictor(GenerateReturn):
    def __init__(self, config, T_max):
        config = deepcopy(config)
        super().__init__(config, T_max)
        baseline_rng = torch.get_rng_state()
        self.research = config.get("research", {})
        self.market_kind = self.research.get("market_conditioning")
        if self.market_kind == "route":
            moe = self.loadings.fusion.moe
            moe.gate = MarketConditionedGate(moe.gate, moe.num_experts)
        elif self.market_kind == "direct":
            # Controls for adding market information/capacity without changing
            # routing. Multiplicative stock interaction can change stock ranks.
            self.market_modulation = nn.Sequential(
                nn.Linear(MARKET_DIM, 16), nn.LayerNorm(16), nn.GELU(),
                nn.Linear(16, self.vq_embed_dim),
            )
            nn.init.zeros_(self.market_modulation[-1].weight)
            nn.init.zeros_(self.market_modulation[-1].bias)
        elif self.market_kind is not None:
            raise ValueError(f"Unknown market conditioning: {self.market_kind}")
        if self.research.get("prior_gate"):
            self.prior_gate = nn.Sequential(
                nn.Linear(self.vq_embed_dim, 32), nn.GELU(),
                nn.Linear(32, self.num_prior_factors),
            )
            nn.init.zeros_(self.prior_gate[-1].weight)
            nn.init.zeros_(self.prior_gate[-1].bias)
        horizons = self.research.get("auxiliary_horizons", [])
        if any(h not in range(2, 11) or h == self.target_index + 1 for h in horizons):
            raise ValueError("Auxiliary horizons must be distinct nontrivial labels apart from the main target")
        if len(horizons) != len(set(horizons)):
            raise ValueError("Duplicate auxiliary horizon")
        if self.research.get("multihorizon_weight", 0):
            if not horizons:
                raise ValueError("Multihorizon supervision requires auxiliary horizons")
            self.horizon_head = nn.Linear(2 * self.vq_embed_dim + self.num_prior_factors, len(horizons))
            nn.init.zeros_(self.horizon_head.weight)
            nn.init.zeros_(self.horizon_head.bias)
        self._auxiliary_prediction = None
        self._prior_gate_values = None
        # Keep the shared dropout/noise RNG stream unchanged by constructing
        # optional branches; existing parameters were already initialized.
        torch.set_rng_state(baseline_rng)

    def forward(self, feature, prior_factor, market_feature=None, frozen_h=None, frozen_zq=None):
        if (frozen_h is None) != (frozen_zq is None):
            raise ValueError("Cached continuous and quantized representations must be supplied together")
        if frozen_h is None:
            normalized = self.revin(feature, mode="norm")
            h = self.encoder(normalized)
            z_q = self.quantizer(h)[0]
        else:
            h, z_q = frozen_h.detach(), frozen_zq.detach()
            expected = (feature.shape[0], self.vq_embed_dim)
            if tuple(h.shape) != expected or tuple(z_q.shape) != expected:
                raise ValueError(f"Cached Stage 1 shape mismatch; expected {expected}")
        z_stage2 = self.build_stage2_latent(h, z_q)
        market = None
        if self.market_kind:
            if market_feature is None or tuple(market_feature.shape) != (feature.shape[0], feature.shape[1], MARKET_DIM):
                raise ValueError("A complete, aligned market63 window is required")
            # CanonicalSampler supplies a common market calendar for all stocks.
            market = market_feature[0, -1, :]
        if self.market_kind == "route":
            gate = self.loadings.fusion.moe.gate
            gate.condition = market
            try:
                alpha, beta_p, beta_l, auxiliary = self.loadings(feature, z_stage2)
            finally:
                gate.condition = None
        else:
            alpha, beta_p, beta_l, auxiliary = self.loadings(feature, z_stage2)
        normalized_prior = self.z_prior_norm(prior_factor)
        self._prior_gate_values = None
        if self.research.get("prior_gate"):
            self._prior_gate_values = 2 * torch.sigmoid(self.prior_gate(z_stage2))
            beta_p = beta_p * self._prior_gate_values
        latent_values = self.latent_value_head(z_stage2)
        prediction = self.return_predictor(alpha, beta_p, beta_l, normalized_prior, latent_values)
        if self.market_kind == "direct":
            prediction = prediction + (z_stage2 * self.market_modulation(market)).sum(-1) / self.vq_embed_dim ** 0.5
        self._auxiliary_prediction = None
        if hasattr(self, "horizon_head"):
            self._auxiliary_prediction = self.horizon_head(torch.cat((beta_p, beta_l, z_stage2), dim=-1))
        return prediction, beta_p, beta_l, z_stage2, softcap_log1p(auxiliary, self.aux_imp)

    @staticmethod
    def batch_parts(batch):
        if isinstance(batch, dict):
            return unpack_batch(batch["canonical"].float()), batch["frozen_h"], batch["frozen_zq"]
        return unpack_batch(batch.float()), None, None

    def batch_prediction(self, batch):
        parts, h, z_q = self.batch_parts(batch)
        result = self(parts.stock_feature, parts.prior_factor, parts.market_feature, h, z_q)
        return parts, result

    def training_loss(self, parts, result):
        prediction, _, _, _, auxiliary = result
        target = parts.target(self.target_index + 1)
        mse = masked_mse(prediction, target)
        terms = {"mse": mse, "auxiliary": auxiliary}
        total = mse + self.aux_weight * auxiliary
        for name, function in (("ic", ic_objective), ("pairwise", pairwise_objective)):
            weight = self.research.get(f"{name}_weight", 0)
            if weight:
                terms[name] = function(prediction, target)
                total = total + weight * terms[name]
        weight = self.research.get("multihorizon_weight", 0)
        if weight:
            targets = torch.stack([parts.target(h) for h in self.research["auxiliary_horizons"]], dim=-1)
            terms["multihorizon"] = masked_mse(self._auxiliary_prediction, targets)
            total = total + weight * terms["multihorizon"]
        weight = self.research.get("prior_gate_penalty", 0)
        if weight:
            terms["prior_gate"] = self._prior_gate_values.mean()
            total = total + weight * terms["prior_gate"]
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite research training loss")
        return total, terms

    def training_step(self, batch, batch_idx):
        parts, result = self.batch_prediction(batch)
        loss, terms = self.training_loss(parts, result)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=len(result[0]))
        self.log_dict({f"train_{name}_loss": value for name, value in terms.items()},
                      on_step=False, on_epoch=True, batch_size=len(result[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        parts, result = self.batch_prediction(batch)
        prediction, _, _, _, auxiliary = result
        mse = masked_mse(prediction, parts.target(self.target_index + 1))
        loss = mse + self.aux_weight * auxiliary
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(prediction))
        self.log("val_mse_loss", mse, on_step=False, on_epoch=True, batch_size=len(prediction))
        from trace_vq.evaluation.inference import calc_ic
        ic, ric = calc_ic(prediction.detach().cpu().numpy(), parts.target(self.target_index + 1).cpu().numpy())
        self.ic.append(ic)
        self.ric.append(ric)
        return loss
