"""Use frozen VQ distortion to correct the latent consumed by Stage 2.

The adapter subclasses Linear directly to retain the original checkpoint keys
(`weight`, `bias`) and consumes the same initialization RNG before zeroing.
"""
import torch
from torch import nn


def quantization_error(h_batch, z_q):
    """Detached per-sample mean squared quantization distortion [N, 1]."""
    return torch.mean(
        (h_batch.detach() - z_q.detach()) ** 2,
        dim=-1,
        keepdim=True,
    )


class QuantizationConfidenceAdapter(nn.Linear):
    """Zero-initialized Linear(1, latent_dim); returns the latent correction."""

    def __init__(self, latent_dim):
        super().__init__(1, latent_dim)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)
