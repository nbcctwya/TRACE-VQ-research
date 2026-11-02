"""Original HVQ experiment 025 encoder for TRACE-VQ's w/o TA ablation.

Module names and construction order retain the historical checkpoint keys.
"""

from torch import nn

from .encoder import CrossAssetTransformerEncoder


class FeatureExtractor(nn.Module):
    def __init__(self, n_feature, hidden_dim):
        super().__init__()
        self.n_feature = n_feature
        self.hidden_dim = hidden_dim
        self.normalize = nn.LayerNorm(n_feature)
        self.linear = nn.Linear(n_feature, n_feature)
        self.leakyrelu = nn.LeakyReLU()
        self.get_h = nn.GRU(n_feature, hidden_dim, batch_first=True)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.leakyrelu(x)
        _, h_n = self.get_h(x)
        return h_n.squeeze(0)


class GRUSpatialEncoder(nn.Module):
    """Per-stock GRU followed by the original cross-asset Transformer."""

    def __init__(self, input_features_C, T_window, gru_hidden_size,
                 num_transformer_heads, num_transformer_layers, final_embed_dim_d):
        super().__init__()
        self.T_window = T_window
        self.C = input_features_C
        self.gru_hidden_size = gru_hidden_size
        self.temporal_extractor = FeatureExtractor(input_features_C, gru_hidden_size)
        self.cross_asset_transformer = CrossAssetTransformerEncoder(
            embed_dim=gru_hidden_size,
            num_heads=num_transformer_heads,
            num_layers=num_transformer_layers,
            d_out=final_embed_dim_d,
        )

    def forward(self, x_batch):
        temporal_summaries = self.temporal_extractor(x_batch)
        return self.cross_asset_transformer(temporal_summaries)
