"""Second research screen: predefined objectives, routing, and code history."""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

from trace_vq.models.screening import ScreeningPredictor
from trace_vq.models.experts.shared_routed import FactorGatedMoE


NEXT_VARIANTS = {
    'baseline': {},
    'pairwise_reference': {'pairwise_weight': 0.1},
    'ranknet': {'ranknet_weight': 0.1},
    'top_ranknet': {'ranknet_weight': 0.1, 'top_pair_boost': 2.0, 'topk': 30},
    'turnover_control': {'pairwise_weight': 0.1, 'paired_days': True, 'turnover_weight': 0.0},
    'turnover': {'pairwise_weight': 0.1, 'paired_days': True, 'turnover_weight': 0.05},
    'soft_route': {'soft_route': True},
    'soft_market': {'soft_route': True, 'market_conditioning': 'route'},
    'history_mean': {'history': 'mean', 'history_days': 20},
    'history_gru': {'history': 'gru', 'history_days': 20, 'history_hidden': 32},
    'rvq_256x256': {'quantizer_kind': 'rvq', 'sizes': [256, 256]},
    'rvq_32x16': {'quantizer_kind': 'rvq', 'sizes': [32, 16]},
    'fsq': {'quantizer_kind': 'fsq', 'levels': [8, 8, 8]},
}
PHASES = (
    ('01_ranking', ['ranknet', 'top_ranknet']),
    ('02_turnover', ['turnover_control', 'turnover']),
    ('03_routing', ['soft_route', 'soft_market']),
    ('04_code_history', ['history_mean', 'history_gru']),
    ('05_residual_vq', ['rvq_256x256', 'rvq_32x16']),
    ('06_fsq', ['fsq']),
)


def ranknet_objective(prediction, target, topk=30, boost=0):
    valid = torch.isfinite(target)
    p, y = prediction[valid], target[valid]
    positive = y[:, None] > y[None, :]
    if not positive.any():
        return prediction.sum() * 0
    penalties = F.softplus(-(p[:, None] - p[None, :]))
    weights = torch.ones_like(penalties)
    if boost:
        head = torch.zeros_like(y, dtype=torch.bool)
        head[y.topk(min(topk, len(y))).indices] = True
        weights = weights + boost * torch.logical_xor(head[:, None], head[None, :])
    return (penalties[positive] * weights[positive]).sum() / weights[positive].sum()


def standardized_turnover(current, previous):
    if len(current) < 2:
        return current.sum() * 0
    def normalize(x):
        x = x - x.mean()
        return x / (x.square().mean() + 1e-4).sqrt()
    return F.smooth_l1_loss(normalize(current), normalize(previous))


class SoftRoutingMoE(FactorGatedMoE):
    """Dense two-expert control with consistent train/eval logits and soft load."""
    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        logits = self.gate(x)
        if self.noisy_gating and train:
            logits = logits + torch.randn_like(logits) * (self.softplus(self.noise(x)) + noise_epsilon)
        probabilities = self.softmax(logits @ self.W_h)
        return probabilities, probabilities.sum(0)


class NextPredictor(ScreeningPredictor):
    def __init__(self, config, T_max):
        super().__init__(deepcopy(config), T_max)
        state = torch.get_rng_state()
        try:
            if self.research.get('soft_route'):
                old = self.loadings.fusion.moe
                cfg = self.config['predictor']
                replacement = SoftRoutingMoE(old.gate_input_size, old.expert_input_size,
                                              hidden_size=cfg['moe_hidden'], num_experts=old.num_experts,
                                              noisy_gating=old.noisy_gating, k=old.k,
                                              use_shared_expert=old.use_shared_expert)
                replacement.gate = old.gate
                replacement.load_state_dict(old.state_dict(), strict=True)
                self.loadings.fusion.moe = replacement
            kind = self.research.get('history')
            if kind:
                self.history_norm = nn.LayerNorm(self.vq_embed_dim)
                if kind == 'gru':
                    hidden = self.research['history_hidden']
                    self.history_gru = nn.GRUCell(self.vq_embed_dim, hidden)
                else:
                    hidden = self.vq_embed_dim
                self.history_head = nn.Linear(hidden, self.vq_embed_dim)
                nn.init.zeros_(self.history_head.weight)
                nn.init.zeros_(self.history_head.bias)
        finally:
            torch.set_rng_state(state)
        self._history_context = None

    def build_stage2_latent(self, h, zq):
        latent = super().build_stage2_latent(h, zq)
        if not self.research.get('history'):
            return latent
        if self._history_context is None:
            raise ValueError('Code history must be supplied as aligned past-only inputs')
        values, mask = self._history_context
        values = self.history_norm(values.detach())
        mask = mask.bool()
        if self.research['history'] == 'mean':
            encoded = (values * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        else:
            encoded = values.new_zeros(len(values), self.research['history_hidden'])
            for step in range(values.shape[1]):
                proposed = self.history_gru(values[:, step], encoded)
                encoded = torch.where(mask[:, step, None], proposed, encoded)
        return latent + self.history_head(encoded) * mask.any(1, keepdim=True)

    def batch_prediction(self, batch):
        if self.research.get('history'):
            self._history_context = (batch['code_history'], batch['history_mask'])
        try:
            return super().batch_prediction(batch)
        finally:
            self._history_context = None

    def training_loss(self, parts, result):
        total, terms = super().training_loss(parts, result)
        if self.research.get('ranknet_weight'):
            value = ranknet_objective(result[0], parts.target(self.target_index + 1),
                                      self.research.get('topk', 30), self.research.get('top_pair_boost', 0))
            terms['ranknet'] = value
            total = total + self.research['ranknet_weight'] * value
        return total, terms

    def training_step(self, batch, batch_idx):
        parts, result = self.batch_prediction(batch)
        total, terms = self.training_loss(parts, result)
        if self.research.get('paired_days') and batch['align_current'].numel() > 1:
            previous = {name: batch['prev_' + name] for name in ('canonical', 'frozen_h', 'frozen_zq')}
            previous_score = self.batch_prediction(previous)[1][0]
            value = standardized_turnover(result[0][batch['align_current'].long()],
                                          previous_score[batch['align_previous'].long()])
            terms['turnover'] = value
            total = total + self.research.get('turnover_weight', 0) * value
        if not torch.isfinite(total):
            raise FloatingPointError('Non-finite next-screen loss')
        self.log('train_loss', total, on_step=False, on_epoch=True, batch_size=len(result[0]))
        self.log_dict({f'train_{key}_loss': value for key, value in terms.items()},
                      on_step=False, on_epoch=True, batch_size=len(result[0]))
        return total
