"""Checklist batches03/04; opt-in additions to the existing frozen predictors."""
from copy import deepcopy

import torch
from torch import nn

from trace_vq.data.fsq_history import FEATURE_DIMS
from trace_vq.models.batch01 import CoordinateFSQ
from trace_vq.models.experts.routing_research import ResearchMoE
from trace_vq.models.screening import ScreeningPredictor

BATCH03={
    "code_delta":dict(item="C1",feature="code_delta"),
    "code_dwell":dict(item="C2",feature="code_dwell"),
    "code_trans":dict(item="C3",feature="code_trans"),
    "code_vol":dict(item="C4",feature="code_vol"),
    "fsq_gram2":dict(item="K4",feature="fsq_gram2",hash_bins=4096,embedding_dim=32),
    "fsq_gram3":dict(item="K4",feature="fsq_gram3",hash_bins=4096,embedding_dim=32),
    "code_hist_state":dict(item="D1",feature="code_hist_state"),
    "proto_memory":dict(item="J20",feature="proto_memory",prototype_decay=.99),
}
BATCH03={name:dict(batch=3,family="fsq",quantizer_kind="fsq",levels=[8,8,8],**value) for name,value in BATCH03.items()}
BATCH04={
    "moe_biasbal":dict(item="J11",routing="bias_balance",load_ema_decay=.99,bias_step=.001),
    "expert_choice":dict(item="J13",routing="expert_choice",capacity_factor=1),
    "moe_finegrained8":dict(item="J12",routing="finegrained",experts=8,expert_width=16,active_experts=4),
    "moe_finegrained16":dict(item="J12",routing="finegrained",experts=16,expert_width=8,active_experts=8),
    "soft_moe":dict(item="J14",routing="soft_slots",slots_per_expert=4),
    "fsq_sort_route":dict(item="B7",routing="fsq_sort",family="fsq",quantizer_kind="fsq",levels=[8,8,8],
                           route_dimension=0,route_cutpoints=[3,5]),
    "expert_k_all":dict(item="G6",routing="all_experts"),
}
BATCH04={name:dict(batch=4,**({"family":"vq"}|value)) for name,value in BATCH04.items()}
VARIANTS=BATCH03|BATCH04


class HistogramGate(nn.Module):
    def __init__(self,base,experts):
        super().__init__()
        self.base=base
        self.projection=nn.Linear(27,experts)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.condition=None

    def forward(self,z):
        if self.condition is None: raise ValueError("Same-day FSQ histogram required")
        return self.base(z)+self.projection(self.condition)


class Batch34Predictor(ScreeningPredictor):
    def __init__(self,config,T_max):
        original=deepcopy(config)
        settings=original["research"]
        variant=settings["variant"]
        # Preserve the shared/native initialization before replacing routed
        # experts; the effective configuration still records actual dimensions.
        construction=deepcopy(original)
        if settings.get("routing") in ("finegrained","fsq_sort"):
            construction["predictor"]["n_expert"]=2
            construction["predictor"]["k"]=1
        super().__init__(construction,T_max)
        self.variant=variant;self.config["research"]=deepcopy(settings)
        self.config["predictor"]["n_expert"]=original["predictor"]["n_expert"]
        self.config["predictor"]["k"]=original["predictor"]["k"]
        state=torch.get_rng_state()
        try:
            if settings["family"]=="fsq": self.quantizer=CoordinateFSQ.from_frozen(self.quantizer)
            if variant in FEATURE_DIMS:
                dimension=self.vq_embed_dim if variant=="proto_memory" else FEATURE_DIMS[variant]
                self.context_projection=nn.Linear(dimension,self.vq_embed_dim)
                nn.init.zeros_(self.context_projection.weight);nn.init.zeros_(self.context_projection.bias)
                self.context_norm=nn.LayerNorm(dimension) if variant=="proto_memory" else nn.Identity()
            elif variant.startswith("fsq_gram"):
                self.memory_embedding=nn.Embedding(settings["hash_bins"],settings["embedding_dim"],padding_idx=0)
                self.memory_projection=nn.Linear(settings["embedding_dim"],self.vq_embed_dim)
                self.memory_gate=nn.Linear(self.vq_embed_dim,self.vq_embed_dim)
                for module in (self.memory_projection,self.memory_gate):
                    nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
            elif variant=="code_hist_state":
                moe=self.loadings.fusion.moe
                moe.gate=HistogramGate(moe.gate,moe.num_experts)
            if settings["batch"]==4:
                self.loadings.fusion.moe=ResearchMoE(self.loadings.fusion.moe,settings)
        finally:
            torch.set_rng_state(state)
        self.context=None

    def build_stage2_latent(self,h,zq):
        latent=super().build_stage2_latent(h,zq)
        if self.variant=="fsq_sort_route":
            q=self.quantizer.coordinates(h.detach())[1]
            coordinate=q[:,self.research["route_dimension"]]
            lo,hi=self.research["route_cutpoints"]
            self.loadings.fusion.moe.route_ids=(coordinate>=lo).long()+(coordinate>=hi).long()
        if self.variant in FEATURE_DIMS:
            if self.context is None: raise ValueError("Aligned past FSQ context required")
            adjustment=self.context_projection(self.context_norm(self.context["context_feature"].detach()))
            if self.variant=="proto_memory": adjustment=adjustment*self.context["context_mask"]
            latent=latent+adjustment
        elif self.variant.startswith("fsq_gram"):
            if self.context is None: raise ValueError("Past FSQ n-grams required")
            ids=self.context["gram_id"].long().squeeze(-1)
            memory=self.memory_projection(self.memory_embedding(ids))
            latent=latent+self.memory_gate(h.detach()).sigmoid()*memory*(ids!=0)[:,None]
        return latent

    def batch_prediction(self,batch):
        self.context=batch if isinstance(batch,dict) else None
        gate=self.loadings.fusion.moe.gate if self.variant=="code_hist_state" else None
        if gate is not None:
            values=batch["state_hist"]
            if not torch.equal(values,values[:1].expand_as(values)): raise ValueError("Histogram must match the whole date")
            gate.condition=values[0]
        try:
            return super().batch_prediction(batch)
        finally:
            self.context=None
            if gate is not None: gate.condition=None
            if self.variant=="fsq_sort_route": self.loadings.fusion.moe.route_ids=None
