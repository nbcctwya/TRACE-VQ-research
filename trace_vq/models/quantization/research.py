"""Explicit research quantizers; default construction is the historical VQ."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .codebook import VectorQuantiser


class ResidualVectorQuantiser(nn.Module):
    def __init__(self, sizes, embed_dim, beta, **kwargs):
        super().__init__()
        if len(sizes) != 2 or min(sizes) < 2:
            raise ValueError('Residual experiment requires two nontrivial codebooks')
        self.sizes = tuple(sizes)
        self.num_embed = sum(sizes)
        self.combined_vocab_size = math.prod(sizes)
        self.coarse = VectorQuantiser(sizes[0], embed_dim, beta, **kwargs)
        self.fine = VectorQuantiser(sizes[1], embed_dim, beta, **kwargs)

    def forward(self, h):
        coarse, loss_c, (_, enc_c, idx_c) = self.coarse(h)
        fine, loss_f, (_, enc_f, idx_f) = self.fine(h - coarse.detach())
        quantized = h + (coarse + fine - h).detach()
        indices = idx_c * self.sizes[1] + idx_f
        probabilities = torch.unique(indices, return_counts=True)[1].float() / len(indices)
        perplexity = (-(probabilities * probabilities.log()).sum()).exp()
        return quantized, (loss_c + loss_f) / 2, (perplexity, torch.cat((enc_c, enc_f), -1), indices)


class FiniteScalarQuantiser(nn.Module):
    """Learned projections around a fixed bounded scalar product grid."""
    def __init__(self, levels, embed_dim):
        super().__init__()
        if min(levels) < 2:
            raise ValueError('Every scalar grid needs at least two levels')
        self.num_embed = math.prod(levels)
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.long))
        self.register_buffer('basis', torch.tensor([math.prod(levels[:i]) for i in range(len(levels))]))
        self.to_grid = nn.Linear(embed_dim, len(levels))
        self.from_grid = nn.Linear(len(levels), embed_dim)

    def forward(self, h):
        continuous = (self.to_grid(h).tanh() + 1) / 2 * (self.levels - 1)
        integers = continuous.round()
        straight_through = continuous + (integers - continuous).detach()
        quantized = self.from_grid(2 * straight_through / (self.levels - 1) - 1)
        indices = (integers.long() * self.basis).sum(-1)
        encodings = F.one_hot(indices, self.num_embed).to(h.dtype)
        probabilities = encodings.mean(0)
        perplexity = (-(probabilities * (probabilities + 1e-10).log()).sum()).exp()
        return quantized, quantized.sum() * 0, (perplexity, encodings, indices)


def build_quantizer(config):
    quant = config['quantizer']
    kwargs = dict(distance=quant['distance'], anchor=quant['anchor'],
                  first_batch=quant['first_batch'], contras_loss=quant['contras_loss'])
    # Consume historical RNG so every common encoder/decoder initialization agrees.
    legacy = VectorQuantiser(config['num_embed'], config['vq_embed_dim'], quant['commit_weight'], **kwargs)
    kind = quant.get('research_kind', 'legacy')
    if kind == 'legacy':
        return legacy
    state = torch.get_rng_state()
    try:
        if kind == 'rvq':
            return ResidualVectorQuantiser(quant['research_sizes'], config['vq_embed_dim'],
                                           quant['commit_weight'], **kwargs)
        if kind == 'fsq':
            result = FiniteScalarQuantiser(quant['research_levels'], config['vq_embed_dim'])
            if result.num_embed != config['num_embed']:
                raise ValueError('FSQ effective vocabulary differs from declared K')
            return result
        raise ValueError(f'Unknown research quantizer: {kind}')
    finally:
        torch.set_rng_state(state)
