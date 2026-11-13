"""Opt-in Stage1 quantizers, outside the legacy cache fingerprint directory.

No legacy quantizer or imported weight/cache is changed. Large product grids
return integer IDs, never product-vocabulary one-hot tensors.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from trace_vq.models.quantization.research import FiniteScalarQuantiser


def quantizer_result(z,loss,ids):
    counts=torch.unique(ids,return_counts=True)[1].to(z.dtype);p=counts/counts.sum()
    perplexity=(-(p*p.log()).sum()).exp()
    return z,loss,(perplexity,z.new_empty((len(z),0)),ids)


def cross_correlation(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    x=x/(x.square().mean(0)+1e-5).sqrt();y=y/(y.square().mean(0)+1e-5).sqrt()
    return x.T@y/max(1,len(x))


def rank_bins(raw,levels):
    """Average tied ranks, same-date quantiles; no labels or fitted thresholds."""
    result=[]
    for j,L in enumerate(levels.tolist()):
        _,inverse,counts=torch.unique(raw[:,j].detach(),sorted=True,return_inverse=True,return_counts=True)
        midpoint=counts.cumsum(0)-counts/2
        result.append((midpoint[inverse]/len(raw)*L).floor().clamp(0,L-1))
    return torch.stack(result,-1).to(raw.dtype)


class ResearchFSQ(FiniteScalarQuantiser):
    def __init__(self,original,settings):
        super().__init__(original.levels.tolist(),original.to_grid.in_features)
        self.load_state_dict(original.state_dict(),strict=True);self.settings=settings;self.kind=settings['quantizer']
        self.last_raw=None
        if self.kind=='scale':self.log_scale=nn.Parameter(torch.zeros(len(self.levels)))

    def forward(self,h):
        raw=self.to_grid(h);self.last_raw=raw
        if self.kind=='scale':raw=raw*self.log_scale.clamp(-5,5).exp()
        if self.kind=='cs':raw=(raw-raw.mean(0))/(raw.var(0,unbiased=False)+1e-5).sqrt()
        scale=self.settings.get('bound_scale',1.)
        u=((raw*scale).tanh()+1)/2*(self.levels-1)
        q=rank_bins(raw,self.levels) if self.kind=='rank' else u.round()
        if self.training and self.kind=='dither':
            noisy=(u+torch.rand_like(u)-.5).clamp_min(0)
            noisy=torch.minimum(noisy,(self.levels-1).to(noisy.dtype));continuous=noisy;q=noisy.round()
        else:continuous=u+(q-u).detach()
        z=self.from_grid(2*continuous/(self.levels-1)-1)
        ids=(q.long()*self.basis).sum(-1)
        return quantizer_result(z,z.sum()*0,ids)


class ResidualFSQ(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.coarse=FiniteScalarQuantiser([3,3,3],dimension)
        self.residual_norm=nn.LayerNorm(dimension);self.fine=FiniteScalarQuantiser([8,8,8],dimension)
        self.num_embed=512;self.combined_vocab_size=27*512
    def forward(self,h):
        coarse,_,ci=self.coarse(h);fine,_,fi=self.fine(self.residual_norm(h-coarse))
        self.last_components=torch.stack((ci[2],fi[2]),-1)
        z=coarse+fine;ids=ci[2]*512+fi[2]
        return quantizer_result(z,z.sum()*0,ids)


class GroupedFSQ(nn.Module):
    def __init__(self,dimension):
        super().__init__();assert dimension%2==0
        self.groups=nn.ModuleList([FiniteScalarQuantiser([8,8,8],dimension//2) for _ in range(2)])
        self.num_embed=512;self.combined_vocab_size=512**2
    def forward(self,h):
        left,right=[module(piece) for module,piece in zip(self.groups,h.chunk(2,-1))]
        self.last_components=torch.stack((left[2][2],right[2][2]),-1)
        z=torch.cat((left[0],right[0]),-1);ids=left[2][2]+512*right[2][2]
        return quantizer_result(z,z.sum()*0,ids)


class HybridFSQ(nn.Module):
    def __init__(self,original,width=4):
        super().__init__();self.grid=original;d=original.to_grid.in_features
        self.continuous=nn.Linear(d,width);self.project=nn.Linear(3+width,d)
        with torch.no_grad():
            self.project.weight.zero_();self.project.weight[:,:3].copy_(original.from_grid.weight)
            self.project.bias.copy_(original.from_grid.bias)
        # A single joint decoder replaces the unused native from_grid.
        del self.grid.from_grid
        self.num_embed=512;self.combined_vocab_size=512
    def forward(self,h):
        u=(self.grid.to_grid(h).tanh()+1)/2*(self.grid.levels-1);q=u.round();straight=u+(q-u).detach()
        z=self.project(torch.cat((2*straight/(self.grid.levels-1)-1,self.continuous(h).tanh()),-1))
        ids=(q.long()*self.grid.basis).sum(-1)
        return quantizer_result(z,z.sum()*0,ids)


def sinkhorn(logits,iterations):
    """Differentiable log-domain scaling to unit rows and N/K column mass."""
    logq=logits-logits.max();n,k=logq.shape
    for _ in range(iterations):
        logq=logq-torch.logsumexp(logq,dim=0,keepdim=True)+math.log(n/k)
        logq=logq-torch.logsumexp(logq,dim=1,keepdim=True)
    return logq.exp()


class SinkhornQuantizer(nn.Module):
    def __init__(self,dimension,settings):
        super().__init__();self.settings=settings;self.codebook=nn.Parameter(torch.randn(512,dimension)*.02)
        self.num_embed=512;self.combined_vocab_size=512;self.last_diagnostics={}
    def assignments(self,h):
        distance=(h.square().sum(-1,keepdim=True)+self.codebook.square().sum(-1)[None]-2*h@self.codebook.T).clamp_min(0)
        return sinkhorn(-distance/self.settings['sinkhorn_epsilon'],self.settings['sinkhorn_iterations'])
    def forward(self,h):
        assignment=self.assignments(h);ids=assignment.argmax(-1);hard=F.embedding(ids,self.codebook)
        soft=assignment@self.codebook;z=soft+(hard-soft).detach()
        loss=F.mse_loss(h,hard.detach())*self.settings['commit_weight']+F.mse_loss(h.detach(),hard)
        with torch.no_grad():
            mass=assignment.sum(0);expected=len(h)/512
            self.last_diagnostics=dict(soft_mass_min=float(mass.min()),soft_mass_max=float(mass.max()),
                soft_column_relative_error=float((mass-expected).abs().max()/expected),hard_used=int(ids.unique().numel()),stocks=len(h))
        return quantizer_result(z,loss,ids)


class BinarySphericalQuantizer(nn.Module):
    """9-bit unit-sphere signs, exact 512-code entropy objective and STE."""
    def __init__(self,original,settings):
        super().__init__();self.settings=settings;self.to_grid=original.to_grid;self.from_grid=original.from_grid
        bits=9;ids=torch.arange(2**bits);basis=2**torch.arange(bits)
        self.register_buffer('basis',basis);self.register_buffer('vertices',(((ids[:,None]//basis)%2)*2-1).float()/math.sqrt(bits))
        self.num_embed=512;self.combined_vocab_size=512
    def spherical(self,h):return F.normalize(self.to_grid(h),dim=-1,eps=1e-6)
    def forward(self,h):
        sphere=self.spherical(h);sign=torch.where(sphere>=0,1.,-1.)/3
        straight=sphere+(sign-sphere).detach();z=self.from_grid(straight)
        ids=((sphere>=0).long()*self.basis).sum(-1)
        logits=2*sphere@self.vertices.T/self.settings['bsq_temperature'];logp=logits.log_softmax(-1);p=logp.exp()
        sample_entropy=-(p*logp).sum(-1).mean();marginal=p.mean(0)
        marginal_entropy=-(marginal*marginal.clamp_min(1e-12).log()).sum()
        loss=self.settings['bsq_entropy_weight']*(sample_entropy-marginal_entropy)
        loss=loss+self.settings['bsq_commit_weight']*F.mse_loss(sphere,sign.detach())
        return quantizer_result(z,loss,ids)


def build_research_quantizer(original,settings):
    kind=settings['quantizer'];dimension=original.to_grid.in_features
    if kind=='native':return original
    if kind=='residual':return ResidualFSQ(dimension)
    if kind=='grouped':return GroupedFSQ(dimension)
    if kind=='hybrid':return HybridFSQ(original,settings['continuous_dim'])
    if kind=='sinkhorn':return SinkhornQuantizer(dimension,settings)
    if kind=='bsq':return BinarySphericalQuantizer(original,settings)
    return ResearchFSQ(original,settings)
