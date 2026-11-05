"""Batch 01: independent attention, confidence and direct-grid experiments.

Stage 1 parameter names and numerical behavior remain unchanged. Coordinate
access is an extension of the existing frozen FSQ, so existing h/z_q caches
retain their original identity and are never rewritten.
"""
from copy import deepcopy

import torch
from torch import nn

from trace_vq.models.confidence.adapter import QuantizationConfidenceAdapter
from trace_vq.models.prediction.heads import LatentValueHead, ReturnPredictor
from trace_vq.models.prediction.loadings import LoadingGenerator
from trace_vq.models.quantization.research import FiniteScalarQuantiser
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.prediction import softcap_log1p


VARIANTS = {
    "tune800_experts4": {"item": "G1", "family": "vq", "n_expert": 4},
    "tune800_experts8": {"item": "G1", "family": "vq", "n_expert": 8},
    "tune800_aux001": {"item": "G1", "family": "vq", "aux_weight": 0.001},
    "tune800_heads4": {"item": "G1", "family": "vq", "num_heads": 4},
    "gated_attn": {"item": "K1/J2", "family": "vq", "attention_gate": "per_head"},
    "fsq_wo_ca": {"item": "B3", "family": "fsq", "confidence_adapter": False},
    "fsq_ca_dim": {"item": "B2", "family": "fsq", "confidence_input": "abs_u_minus_q"},
    "fsq_grid": {"item": "B1", "family": "fsq", "stage2_latent_dim": 3},
}


class CoordinateFSQ(FiniteScalarQuantiser):
    """Native FSQ forward plus integer-unit pre-round/rounded coordinates."""

    def coordinates(self, h):
        u = (self.to_grid(h).tanh() + 1) / 2 * (self.levels - 1)
        return u, u.round()

    def normalized_coordinates(self, h):
        u, q = self.coordinates(h)
        scale = self.levels - 1
        return 2 * u / scale - 1, 2 * q / scale - 1

    def forward_with_coordinates(self, h):
        # Additional API; preserve the original three-value forward contract.
        result = self(h)
        u, q = self.coordinates(h)
        return result, u, q

    @classmethod
    def from_frozen(cls, original):
        if not isinstance(original, FiniteScalarQuantiser):
            raise TypeError("Coordinate experiments require an existing FSQ Stage 1")
        result = cls(original.levels.tolist(), original.to_grid.in_features)
        result.load_state_dict(original.state_dict(), strict=True)
        result.requires_grad_(False)
        return result.eval()


def copy_matching_parameters(destination, source):
    """Keep every unchanged Stage 2 tensor identical in the 3-factor control."""
    current = destination.state_dict()
    common = {name: value for name, value in source.state_dict().items()
              if name in current and current[name].shape == value.shape}
    destination.load_state_dict(common, strict=False)
    return tuple(common)


class Batch01Predictor(ScreeningPredictor):
    def __init__(self, config, T_max):
        super().__init__(deepcopy(config), T_max)
        self.variant = self.research.get("variant", "baseline")
        self.stage2_latent_dim = self.vq_embed_dim
        state = torch.get_rng_state()
        try:
            if self.variant.startswith("fsq_"):
                self.quantizer = CoordinateFSQ.from_frozen(self.quantizer)
            if self.variant == "gated_attn":
                for layer in self.loadings.temporal_transformer.transformer.layers:
                    layer.head_gate = nn.Linear(layer.d_model, layer.num_heads)
                    nn.init.zeros_(layer.head_gate.weight)
                    nn.init.zeros_(layer.head_gate.bias)
            elif self.variant == "fsq_ca_dim":
                self.quantization_confidence_adapter = nn.Linear(len(self.quantizer.levels), self.vq_embed_dim)
                nn.init.zeros_(self.quantization_confidence_adapter.weight)
                nn.init.zeros_(self.quantization_confidence_adapter.bias)
            elif self.variant == "fsq_grid":
                dimension = len(self.quantizer.levels)
                self.stage2_latent_dim = dimension
                # Stage 1 remains 128-dimensional. Only downstream modules use
                # the normalized grid dimension; compatible tensors are retained.
                downstream = deepcopy(self.config)
                downstream["vqvae"]["vq_embed_dim"] = dimension
                downstream["predictor"]["vq_embed_dim"] = dimension
                downstream["predictor"]["stage2_latent_dim"] = dimension
                loading = LoadingGenerator(downstream)
                self.retained_loading_parameters = copy_matching_parameters(loading, self.loadings)
                self.loadings = loading
                self.latent_value_head = LatentValueHead(dimension, dimension)
                self.return_predictor = ReturnPredictor(self.num_prior_factors, dimension, self.use_prior)
                if self.use_quantization_confidence_adapter:
                    self.quantization_confidence_adapter = QuantizationConfidenceAdapter(dimension)
                self.config["predictor"]["stage2_latent_dim"] = dimension
                self.config["research"]["grid_confidence_error"] = "normalized_coordinate_mse"
        finally:
            # Added construction must not shift the shared training RNG stream.
            torch.set_rng_state(state)

    def build_stage2_latent(self, h, zq):
        if self.variant == "fsq_ca_dim":
            u, q = self.quantizer.coordinates(h.detach())
            return zq.detach() + self.quantization_confidence_adapter((u - q).abs().detach())
        return super().build_stage2_latent(h, zq)

    def forward(self, feature, prior_factor, market_feature=None, frozen_h=None, frozen_zq=None):
        if self.variant != "fsq_grid":
            return super().forward(feature, prior_factor, market_feature, frozen_h, frozen_zq)
        if frozen_h is None:
            h = self.encoder(self.revin(feature, mode="norm"))
        else:
            h = frozen_h.detach()
            if tuple(h.shape) != (feature.shape[0], self.vq_embed_dim):
                raise ValueError("Cached Stage 1 continuous dimension mismatch")
        # This path never calls the frozen from_grid projection.
        continuous, rounded = self.quantizer.normalized_coordinates(h.detach())
        latent = rounded.detach()
        if self.use_quantization_confidence_adapter:
            error = (continuous.detach() - latent).square().mean(-1, keepdim=True)
            latent = latent + self.quantization_confidence_adapter(error)
        alpha, beta_p, beta_l, auxiliary = self.loadings(feature, latent)
        prediction = self.return_predictor(alpha, beta_p, beta_l,
                                           self.z_prior_norm(prior_factor), self.latent_value_head(latent))
        return prediction, beta_p, beta_l, latent, softcap_log1p(auxiliary, self.aux_imp)
