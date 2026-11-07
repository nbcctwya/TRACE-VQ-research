"""Checklist 24–27. Frozen native Stage1 and explicit independent Stage2 changes."""
from copy import deepcopy

import torch
from torch import nn

from trace_vq.models.batch01 import CoordinateFSQ
from trace_vq.models.components.decomposition import DLinear
from trace_vq.models.experts.shared_routed import SimpleMLP,SparseDispatcher
from trace_vq.models.prediction.temporal_transformer import TemporalTransformerEncoder
from trace_vq.models.prediction.batch05_temporal import KDASummary,MHCStack
from trace_vq.models.screening import ScreeningPredictor

VARIANTS={
    "kda_temporal":dict(item="K3",family="vq",backend="torch_reference",heads=2,head_dim=32,conv_size=4,readout="last_structure_token",position_encoding="none"),
    "mhc":dict(item="K2",family="vq",streams=2,sinkhorn_iterations=20,alpha_init=.01,residual_diagonal_bias=2),
    "raven_context_experts":dict(item="K8",family="fsq",quantizer_kind="fsq",levels=[8,8,8],windows=[5,10,20],experts=3,active_experts=1,short_tower_structure_token=False),
    "s2_cross_attn":dict(item="J10",family="vq",cross_heads=2,cross_dimension=64,cross_dropout=.1,cross_zero_init=True),
}


class WindowTower(nn.Module):
    def __init__(self,config,window):
        super().__init__();self.window=window
        c=deepcopy(config);p=c['predictor'];p['pred_len']=window;p['vq_embed_dim']=c['vqvae']['vq_embed_dim']
        p['transformer']['prepend_structure_token']=False
        dimension=p['transformer']['d_model'];features=c['vqvae']['num_features']
        self.dlinear=DLinear(window,window,features,p['kernel_size'],p['individual'])
        self.project=nn.Sequential(nn.Linear(features,dimension),nn.GELU())
        self.temporal=TemporalTransformerEncoder(p)

    def forward(self,feature):
        local=self.project(self.dlinear(feature[:,-self.window:]))
        # A learned CLS reads the local window. Full-window FSQ information
        # is reserved for the router, never injected into these local towers.
        dummy=local.new_zeros(len(feature),self.temporal.z_q_dim)
        return self.temporal(local,dummy)


class WindowMoE(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.window_values=None
        d=base.expert_input_size
        base.experts.append(SimpleMLP(d,d,base.hidden_size));base.num_experts=3;base.k=1
        base.gate[-1]=nn.Linear(base.hidden_size,3);base.noise[-1]=nn.Linear(base.hidden_size,3)
        base.W_h=nn.Parameter(torch.eye(3))
        self.value_projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d)) for _ in range(3)])
        self.days=0;self.usage=None

    def reset_diagnostics(self):self.days=0;self.usage=None
    def diagnostics(self):
        return dict(windows=[5,10,20],days=self.days,mean_route_fractions=None if self.usage is None else (self.usage/max(1,self.days)).cpu().tolist(),all_towers_computed=True)

    def forward(self,x,z,loss_coef=1):
        if self.window_values is None:raise ValueError('Aligned 5/10/20-day expert values required')
        b=self.base;gates,load=b.noisy_top_k_gating(z,self.training)
        dispatcher=SparseDispatcher(3,gates)
        positions=dispatcher.dispatch(torch.arange(len(x),device=x.device)[:,None])
        outputs=[expert(projection(value)[index.flatten().long()]) for expert,projection,value,index in zip(b.experts,self.value_projections,self.window_values,positions)]
        result=dispatcher.combine(outputs)
        if b.shared_expert is not None:result=result+b.shared_expert(x)
        auxiliary=(b.cv_squared(gates.sum(0))+b.cv_squared(load))*loss_coef
        usage=(gates>0).float().mean(0).detach();self.usage=usage if self.usage is None else self.usage+usage;self.days+=1
        return result,auxiliary


class Batch05Loadings(nn.Module):
    def __init__(self,base,config,variant):
        super().__init__();self.base=base;self.variant=variant
        if variant=='kda_temporal':base.temporal_transformer=KDASummary(base.temporal_transformer)
        if variant=='mhc':base.temporal_transformer.transformer=MHCStack(base.temporal_transformer.transformer)
        if variant=='raven_context_experts':
            self.towers=nn.ModuleList([WindowTower(config,w) for w in (5,10)])
            base.fusion.moe=WindowMoE(base.fusion.moe)
        if variant=='s2_cross_attn':
            d=config['predictor']['transformer']['d_model'];heads=config['research']['cross_heads']
            self.cross_norm=nn.LayerNorm(d);self.cross=nn.MultiheadAttention(d,heads,dropout=.1,batch_first=True)
            nn.init.zeros_(self.cross.out_proj.weight);nn.init.zeros_(self.cross.out_proj.bias)

    @property
    def fusion(self):return self.base.fusion

    def forward(self,feature,z):
        b=self.base;h=b.temporal_transformer(b.dim_projection(b.dliner(feature)),z)
        if self.variant=='s2_cross_attn':
            normalized=self.cross_norm(h).unsqueeze(0)
            h=h+self.cross(normalized,normalized,normalized,need_weights=False)[0].squeeze(0)
        if self.variant=='raven_context_experts':
            moe=b.fusion.moe;moe.window_values=[tower(feature) for tower in self.towers]+[h]
            try:return b.fusion(h,z)
            finally:moe.window_values=None
        return b.fusion(h,z)


class Batch05Predictor(ScreeningPredictor):
    def __init__(self,config,T_max):
        original=deepcopy(config);construction=deepcopy(config);variant=config['research']['variant']
        if variant=='raven_context_experts':construction['predictor'].update(n_expert=2,k=1)
        super().__init__(construction,T_max)
        self.config['predictor']['n_expert']=original['predictor']['n_expert']
        state=torch.get_rng_state()
        try:
            if VARIANTS[variant]['family']=='fsq':self.quantizer=CoordinateFSQ.from_frozen(self.quantizer)
            self.loadings=Batch05Loadings(self.loadings,self.config,variant)
        finally:torch.set_rng_state(state)
