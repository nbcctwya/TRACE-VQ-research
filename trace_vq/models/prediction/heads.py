"""Latent factor values and additive prior/latent return prediction."""
import torch
from torch import nn

class LatentValueHead(nn.Module):
    def __init__(self, d_latent, K):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_latent, K),
            nn.LayerNorm(K),
            nn.GELU(),
            nn.Linear(K, K)
        )
    def forward(self, z_q):
        return self.head(z_q)    # (B, K)

class ReturnPredictor(nn.Module):
    def __init__(self, num_prior, num_latent, use_prior=True):
        super().__init__()
        self.num_prior = num_prior
        self.num_latent = num_latent
        self.use_prior = use_prior
        
    def forward(self, alpha, beta_p, beta_l, f_prior, f_latent):
        prior_term = (beta_p * f_prior).sum(dim=1)
        latent_term = (beta_l * f_latent).sum(dim=1)  # elementwise (B, K)

        combined = torch.cat([prior_term.unsqueeze(1), latent_term.unsqueeze(1)], dim=1)
        # intercept_term = self.final_layer(combined).squeeze(-1)
        
        if self.use_prior:  
            output = alpha + prior_term + latent_term # + intercept_term
        else:
            output = alpha + latent_term

        return output
