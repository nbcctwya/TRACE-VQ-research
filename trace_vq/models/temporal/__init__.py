"""TRACE-VQ temporal components."""

from .encoder import SpatialEncoder
from .gru import GRUSpatialEncoder


def build_spatial_encoder(encoder_config, **kwargs):
    """Select the historical encoder without changing either parameter layout."""
    kind = encoder_config.get("type", "temporal-attention")
    if kind == "gru":
        return GRUSpatialEncoder(**kwargs)
    return SpatialEncoder(
        **kwargs,
        encoder_type=kind,
        temporal_dropout=encoder_config.get("temporal_dropout", 0.1),
    )
