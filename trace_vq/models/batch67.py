"""Checklist 28–38: frozen FSQ, isolated grid heads and code conditioning.

All variants retain the native 128-factor downstream interface. New training
penalties are excluded from checkpoint selection. Historical retrieval inputs
must be supplied by the separately audited, maturity-aware dataset.
"""
from copy import deepcopy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from trace_vq.models.batch01 import CoordinateFSQ
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.prediction import softcap_log1p

BATCH06 = {
    "fsq_soft": dict(item="B4", grid_readout="normalized_q_and_u", continuous_zero_init=True),
    "fsq_level_emb": dict(item="B5", embedding_dim=128, embedding_init="native_from_grid_additive"),
    "fsq_cross": dict(item="B6", embedding_dim=128, pair_tables=3, pair_zero_init=True),
    "fsq_grid_smooth": dict(item="B8", grid_smooth_weight=.01, adjacency="six_axis_neighbors", ca_condition="same_vector_on_both_ends"),
    "fsq_shrink": dict(item="B9", shrink_weight=.01, shrink_tau=20., parent_width=2, unseen_code="parent_intercept"),
    "fsq_code_aug": dict(item="B10", boundary_width=.1, flip_probability=.1, augmentation="training_only_per_coordinate"),
    "kan_head": dict(item="J15", kan_width=16, kan_grid_intervals=5, kan_degree=3, kan_hidden_activation="tanh", value_input="normalized_q_only"),
}
BATCH07 = {
    "code_hypergraph": dict(item="J9", hyperedges="same_or_manhattan_one_code", message_layers=1, message_zero_init=True),
    "hyper_loading": dict(item="J16", code_embedding_dim=16, adapter_rank=4, generated="low_rank_output_matrix", adapter_zero_init=True),
    "film_code": dict(item="J17", code_embedding_dim=16, conditioning="per_temporal_layer_input_film", readout="learned_cls", film_zero_init=True),
    "hopfield_retrieval": dict(item="J18", candidates=16, bucket_capacity=8, key_dim=32, inverse_temperature=4., retrieval_steps=1,
        candidate_policy="most_recent_same_or_axis_neighbor_code", label_horizon=5, maturity="source_day_plus_5_strictly_before_query_day",
        memory_updates="causal_across_train_valid_test", value="processed_main_target", retrieval_zero_init=True),
}
VARIANTS = {**{v: dict(batch=6, family="fsq", **s) for v,s in BATCH06.items()},
            **{v: dict(batch=7, family="fsq", **s) for v,s in BATCH07.items()}}


def grid_codes(q):
    return (q.long() * q.new_tensor([1,8,64],dtype=torch.long)).sum(-1)


def all_grid():
    ids=torch.arange(512)
    return torch.stack((ids%8,ids//8%8,ids//64),-1)


def grid_adjacency():
    q=all_grid()
    return (q[:,None]-q[None,:]).abs().sum(-1)<=1


def augment_coordinates(u,q,probability,width):
    """Flip only the other bin at a nearby half-integer, within the grid."""
    lower=u.floor()
    other=torch.where(q==lower,lower+1,lower)
    eligible=((u-lower-.5).abs()<=width)&(other>=0)&(other<=7)
    flip=(torch.rand_like(u)<probability)&eligible
    return torch.where(flip,other,q)


class GridEmbedding(nn.Module):
    def __init__(self,quantizer,pairs=False):
        super().__init__();d=quantizer.from_grid.out_features
        self.single=nn.ModuleList([nn.Embedding(8,d) for _ in range(3)])
        self.pair=nn.ModuleList([nn.Embedding(64,d) for _ in range(3)]) if pairs else nn.ModuleList()
        with torch.no_grad():
            bins=torch.linspace(-1,1,8)
            for j,table in enumerate(self.single):
                table.weight.copy_(bins[:,None]*quantizer.from_grid.weight[:,j]+quantizer.from_grid.bias/3)
            for table in self.pair:table.weight.zero_()

    def forward(self,q):
        q=q.long();result=sum(table(q[:,j]) for j,table in enumerate(self.single))
        for table,(a,b) in zip(self.pair,((0,1),(0,2),(1,2))):result=result+table(q[:,a]*8+q[:,b])
        return result


class SplineLayer(nn.Module):
    """KAN edge functions: SiLU base plus uniform cubic B-splines (no package)."""
    def __init__(self,inputs,outputs,intervals=5,degree=3):
        super().__init__();self.degree=degree
        self.register_buffer('knots',torch.arange(-degree,intervals+degree+1,dtype=torch.float32)*(2/intervals)-1)
        self.base=nn.Linear(inputs,outputs)
        self.coefficients=nn.Parameter(torch.empty(outputs,inputs,intervals+degree))
        nn.init.normal_(self.coefficients,std=.01)

    def basis(self,x):
        x=x.clamp(-1,1)[...,None];k=self.knots
        result=((x>=k[:-1])&(x<k[1:])).to(x.dtype)
        for p in range(1,self.degree+1):
            result=(x-k[:-(p+1)])/(k[p:-1]-k[:-(p+1)])*result[...,:-1]+(k[p+1:]-x)/(k[p+1:]-k[1:-p])*result[...,1:]
        return result

    def forward(self,x):
        return self.base(F.silu(x))+F.linear(self.basis(x).flatten(1),self.coefficients.flatten(1))


class GridKAN(nn.Module):
    def __init__(self,outputs,settings):
        super().__init__();w=settings['kan_width'];g=settings['kan_grid_intervals'];p=settings['kan_degree']
        self.first=SplineLayer(3,w,g,p);self.second=SplineLayer(w,outputs,g,p)
    def forward(self,q):return self.second(self.first(q).tanh())


class GridHypergraph(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.register_buffer('membership',grid_adjacency())
        self.norm=nn.LayerNorm(dimension);self.output=nn.Linear(dimension,dimension)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def message(self,h,code):
        incidence=self.membership[code].to(h.dtype)
        edges=incidence.T@h/incidence.sum(0).clamp_min(1)[:,None]
        return incidence@edges/incidence.sum(-1).clamp_min(1)[:,None]
    def forward(self,h,code):return h+self.output(self.norm(self.message(h,code)))


class CodeFiLM(nn.Module):
    def __init__(self,temporal,settings):
        super().__init__();self.base=temporal
        temporal.prepend_structure_token=False
        self.embedding=nn.Embedding(512,settings['code_embedding_dim'])
        self.film=nn.ModuleList([nn.Linear(settings['code_embedding_dim'],2*temporal.d_model) for _ in temporal.transformer.layers])
        for layer in self.film:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
    def forward(self,x,code):
        b=self.base
        if x.shape[-1]!=b.d_model:x=b.input_proj(x)
        x=torch.cat((b.cls_token.expand(len(x),-1,-1),x),1);embedding=self.embedding(code)
        for layer,condition in zip(b.transformer.layers,self.film):
            gamma,beta=condition(embedding).chunk(2,-1)
            x=layer(x*(1+gamma[:,None])+beta[:,None])
        if b.transformer.norm is not None:x=b.transformer.norm(x)
        return x[:,0]


class CodeLoadingAdapter(nn.Module):
    def __init__(self,dimension,prior,latent,settings):
        super().__init__();self.prior=prior;self.rank=settings['adapter_rank']
        self.embedding=nn.Embedding(512,settings['code_embedding_dim'])
        self.input=nn.Linear(dimension,self.rank,bias=False)
        self.generate=nn.Linear(settings['code_embedding_dim'],self.rank*(prior+latent))
        nn.init.zeros_(self.generate.weight);nn.init.zeros_(self.generate.bias)
    def forward(self,h,code):
        weights=self.generate(self.embedding(code)).reshape(len(h),self.rank,-1)
        delta=(self.input(h)[:,:,None]*weights).sum(1)/self.rank**.5
        return delta[:,:self.prior],delta[:,self.prior:]


class HistoricalHopfield(nn.Module):
    def __init__(self,dimension,settings):
        super().__init__();k=settings['key_dim'];self.beta=settings['inverse_temperature']
        self.query=nn.Linear(dimension,k,bias=False);self.key=nn.Linear(dimension,k,bias=False)
        self.output=nn.Linear(dimension+1,dimension)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def retrieve(self,h,keys,values,mask):
        mask=mask.bool();query=F.normalize(self.query(h),dim=-1);key=F.normalize(self.key(keys),dim=-1)
        logits=(key*query[:,None]).sum(-1)*self.beta
        weights=logits.masked_fill(~mask,-1e9).softmax(-1)*mask
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        memory=torch.cat((keys,values[...,None]),-1)
        return (weights[...,None]*memory).sum(1),weights
    def forward(self,h,keys,values,mask):
        context,_=self.retrieve(h,keys,values,mask)
        return self.output(context)*mask.any(-1,keepdim=True)


class Batch67Predictor(ScreeningPredictor):
    def __init__(self,config,T_max):
        super().__init__(deepcopy(config),T_max);self.variant=self.research['variant'];s=self.research
        if self.variant not in VARIANTS:raise ValueError(self.variant)
        state=torch.get_rng_state()
        try:
            self.quantizer=CoordinateFSQ.from_frozen(self.quantizer)
            d=self.vq_embed_dim;self.grid_penalty=None;self.memory=None
            if self.variant=='fsq_soft':
                self.grid_projection=nn.Linear(6,d)
                with torch.no_grad():
                    self.grid_projection.weight.zero_();self.grid_projection.weight[:,:3].copy_(self.quantizer.from_grid.weight)
                    self.grid_projection.bias.copy_(self.quantizer.from_grid.bias)
            if self.variant in ('fsq_level_emb','fsq_cross'):self.grid_embedding=GridEmbedding(self.quantizer,self.variant=='fsq_cross')
            if self.variant=='fsq_shrink':
                counts=json.loads(Path(s['statistics_file']).read_text())['train_code_counts'] if s.get('statistics_file') else [1]*512
                self.register_buffer('code_counts',torch.tensor(counts,dtype=torch.float32))
                # Parent grid uses base 4, unlike the native base-8 grid.
                self.register_buffer('parents',(all_grid()//2*torch.tensor([1,4,16])).sum(-1))
                self.child_intercept=nn.Embedding(512,1);self.parent_intercept=nn.Embedding(64,1)
                nn.init.zeros_(self.child_intercept.weight);nn.init.zeros_(self.parent_intercept.weight)
            if self.variant=='kan_head':self.latent_value_head=GridKAN(d,s)
            dimension=config['predictor']['transformer']['d_model']
            if self.variant=='code_hypergraph':self.hypergraph=GridHypergraph(dimension)
            if self.variant=='hyper_loading':self.loading_adapter=CodeLoadingAdapter(dimension,self.num_prior_factors,d,s)
            if self.variant=='film_code':
                self.code_temporal=CodeFiLM(self.loadings.temporal_transformer,s)
                # Own the temporal module once; its native structure-token path is removed.
                del self.loadings.temporal_transformer
            if self.variant=='hopfield_retrieval':self.hopfield=HistoricalHopfield(d,s)
        finally:torch.set_rng_state(state)

    def batch_prediction(self,batch):
        self.memory=None
        if self.variant=='hopfield_retrieval':
            if not isinstance(batch,dict):raise ValueError('Causal Hopfield memory is required')
            self.memory=tuple(batch['memory_'+name] for name in ('keys','values','mask'))
        try:return super().batch_prediction(batch)
        finally:self.memory=None

    def forward(self,feature,prior_factor,market_feature=None,frozen_h=None,frozen_zq=None):
        if (frozen_h is None)!=(frozen_zq is None):raise ValueError('Both cached representations are required')
        if frozen_h is None:
            with torch.no_grad():
                h=self.encoder(self.revin(feature,mode='norm'));zq=self.quantizer(h)[0]
        else:h,zq=frozen_h.detach(),frozen_zq.detach()
        u,q=self.quantizer.coordinates(h);q=q.detach()
        if self.variant=='fsq_code_aug' and self.training:
            q=augment_coordinates(u,q,self.research['flip_probability'],self.research['boundary_width'])
            zq=self.quantizer.from_grid(2*q/7-1)
        code=grid_codes(q);normalized=2*q/7-1
        if self.variant=='fsq_soft':zq=self.grid_projection(torch.cat((normalized,2*u/7-1),-1))
        if self.variant in ('fsq_level_emb','fsq_cross'):zq=self.grid_embedding(q)
        # Confidence always uses the unmodified native reconstruction, except
        # code augmentation where all downstream code-dependent inputs change.
        native=frozen_zq if frozen_zq is not None else self.quantizer(h)[0]
        ca_reference=zq if self.variant=='fsq_code_aug' else native
        ca=self.quantization_confidence_adapter(self.quantization_error(h,ca_reference)) if self.use_quantization_confidence_adapter else torch.zeros_like(zq)
        z=zq+ca
        if self.variant=='hopfield_retrieval':
            if self.memory is None:raise ValueError('Maturity-audited memory missing')
            z=z+self.hopfield(h,*self.memory)
        b=self.loadings;temporal_input=b.dim_projection(b.dliner(feature))
        temporal=self.code_temporal(temporal_input,code) if self.variant=='film_code' else b.temporal_transformer(temporal_input,z)
        if self.variant=='code_hypergraph':temporal=self.hypergraph(temporal,code)
        alpha,beta_p,beta_l,auxiliary=b.fusion(temporal,z)
        if self.variant=='hyper_loading':
            old_norm=beta_p.norm()+beta_l.norm();dp,dl=self.loading_adapter(temporal,code)
            beta_p,beta_l=beta_p+dp,beta_l+dl;auxiliary=auxiliary-old_norm+beta_p.norm()+beta_l.norm()
        if self.variant=='fsq_shrink':
            child=self.child_intercept(code).squeeze(-1);parent=self.parent_intercept(self.parents[code]).squeeze(-1)
            alpha=alpha+torch.where(self.code_counts[code]>0,child,parent)
        values=self.latent_value_head(normalized if self.variant=='kan_head' else z)
        prediction=self.return_predictor(alpha,beta_p,beta_l,self.z_prior_norm(prior_factor),values)
        self.grid_penalty=None
        if self.training and self.variant=='fsq_grid_smooth':
            offsets=torch.cat((torch.eye(3,device=q.device),-torch.eye(3,device=q.device)),0)
            neighbor=q[:,None]+offsets;valid=((neighbor>=0)&(neighbor<=7)).all(-1)
            # Both endpoints receive exactly the same, differentiable CA vector.
            neighboring_z=self.quantizer.from_grid(2*neighbor.clamp(0,7)/7-1)+ca[:,None]
            neighboring_values=self.latent_value_head(neighboring_z.flatten(0,1)).view(len(q),6,-1)
            self.grid_penalty=((neighboring_values-values[:,None]).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
        if self.training and self.variant=='fsq_shrink':
            difference=self.child_intercept.weight[:,0]-self.parent_intercept(self.parents).squeeze(-1)
            tau=self.research['shrink_tau'];weight=tau/(self.code_counts+tau)
            self.grid_penalty=(weight*difference.square()).mean()
        return prediction,beta_p,beta_l,z,softcap_log1p(auxiliary,self.aux_imp)

    def training_loss(self,parts,result):
        total,terms=super().training_loss(parts,result)
        weight=self.research.get('grid_smooth_weight',self.research.get('shrink_weight',0))
        if weight:
            terms['grid_regularizer']=self.grid_penalty;total=total+weight*self.grid_penalty
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite grid loss')
        return total,terms
