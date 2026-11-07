"""Small Stage2 KDA and two-stream mHC; no external kernels required.

Independent implementations of the recurrences/constraints described in
https://arxiv.org/abs/2510.26692 and https://arxiv.org/abs/2512.24880.
KDA parameterization was checked against fla/layers/kda.py and its naive
operator. This is a PyTorch reference backend, not a fused FLA benchmark.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def recurrent_kda(q,k,v,log_decay,beta):
    """Per-key-coordinate decay followed by a rank-one delta update."""
    q,k,v,log_decay,beta=[t.float() for t in (q,k,v,log_decay,beta)]
    state=q.new_zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3])
    outputs=[]
    for t in range(q.shape[1]):
        key=k[:,t]
        decayed=state*log_decay[:,t].exp().unsqueeze(-1)
        innovation=(v[:,t]-(key.unsqueeze(-1)*decayed).sum(-2))*beta[:,t].unsqueeze(-1)
        state=decayed+key.unsqueeze(-1)*innovation.unsqueeze(-2)
        outputs.append((q[:,t].unsqueeze(-1)*state).sum(-2)/math.sqrt(q.shape[-1]))
    return torch.stack(outputs,dim=1),state


class TorchKDA(nn.Module):
    def __init__(self,dimension,heads,conv_size=4):
        super().__init__();assert dimension%heads==0
        self.heads,self.width,self.conv_size=heads,dimension//heads,conv_size
        self.qkv=nn.ModuleList([nn.Linear(dimension,dimension,bias=False) for _ in range(3)])
        self.convs=nn.ModuleList([nn.Conv1d(dimension,dimension,conv_size,groups=dimension,bias=False) for _ in range(3)])
        self.forget=nn.Sequential(nn.Linear(dimension,self.width,bias=False),nn.Linear(self.width,dimension,bias=False))
        self.beta=nn.Linear(dimension,heads,bias=False)
        self.A_log=nn.Parameter(torch.empty(heads).uniform_(1,16).log())
        dt=torch.exp(torch.empty(dimension).uniform_(math.log(.001),math.log(.1)))
        self.dt_bias=nn.Parameter(dt+torch.log(-torch.expm1(-dt)))
        self.output_gate=nn.Sequential(nn.Linear(dimension,self.width,bias=False),nn.Linear(self.width,dimension))
        self.norm_weight=nn.Parameter(torch.ones(self.width));self.output=nn.Linear(dimension,dimension,bias=False)

    def forward(self,x):
        values=[]
        for projection,conv in zip(self.qkv,self.convs):
            y=projection(x).transpose(1,2)
            y=F.silu(conv(F.pad(y,(self.conv_size-1,0))).transpose(1,2))
            values.append(y.reshape(*x.shape[:2],self.heads,self.width))
        q,k,v=values;q=F.normalize(q,dim=-1);k=F.normalize(k,dim=-1)
        gate=F.softplus(self.forget(x).float()+self.dt_bias).reshape_as(q)
        log_decay=-self.A_log.exp()[None,None,:,None]*gate
        result,_=recurrent_kda(q,k,v,log_decay,self.beta(x).sigmoid())
        result=result*torch.rsqrt(result.square().mean(-1,keepdim=True)+1e-5)*self.norm_weight
        result=result*self.output_gate(x).reshape_as(result).sigmoid()
        return self.output(result.flatten(-2))


class KDABlock(nn.Module):
    def __init__(self,original):
        super().__init__()
        for name in ("linear1","linear2","norm1","norm2","dropout1","dropout2","dropout_ff"):
            setattr(self,name,getattr(original,name))
        self.activation=original.activation
        self.mixer=TorchKDA(original.d_model,original.num_heads)

    def forward(self,x):
        x=self.norm1(x+self.dropout1(self.mixer(x)))
        y=self.linear2(self.dropout_ff(self.activation(self.linear1(x))))
        return self.norm2(x+self.dropout2(y))


class KDASummary(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base
        self.layers=nn.ModuleList([KDABlock(layer) for layer in base.transformer.layers])
        del base.transformer

    def forward(self,x,z):
        b=self.base
        if x.shape[-1]!=b.d_model:x=b.input_proj(x)
        if z.shape[-1]!=b.d_model:z=b.z_q_proj(z)
        # A first-token readout in a causal recurrence could not see history.
        # Put the structure/readout token last, then read its updated state.
        x=torch.cat((x,z.to(x).unsqueeze(1)),dim=1)
        for layer in self.layers:x=layer(x)
        return x[:,-1]


def sinkhorn(logits,iterations=20):
    log_matrix=logits.float()
    for _ in range(iterations):
        log_matrix=log_matrix-torch.logsumexp(log_matrix,dim=-2,keepdim=True)
        log_matrix=log_matrix-torch.logsumexp(log_matrix,dim=-1,keepdim=True)
    return log_matrix.exp()


class HyperConnection(nn.Module):
    def __init__(self,dimension,streams=2,iterations=20):
        super().__init__();self.streams,self.iterations=streams,iterations
        self.phi=nn.Linear(streams*dimension,streams*2+streams**2,bias=False)
        self.alpha=nn.Parameter(torch.full((3,),.01))
        self.pre_bias=nn.Parameter(torch.zeros(streams));self.post_bias=nn.Parameter(torch.zeros(streams))
        self.residual_bias=nn.Parameter(torch.eye(streams)*2)

    def coefficients(self,x):
        flat=x.flatten(-2);normalized=flat*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-5)
        pre,post,res=self.phi(normalized).split([self.streams,self.streams,self.streams**2],dim=-1)
        pre=(self.alpha[0]*pre+self.pre_bias).sigmoid()
        post=2*(self.alpha[1]*post+self.post_bias).sigmoid()
        residual=sinkhorn(self.alpha[2]*res.reshape(*res.shape[:-1],self.streams,self.streams)+self.residual_bias,self.iterations)
        return pre,post,residual

    def forward(self,x,function,norm):
        pre,post,residual=self.coefficients(x)
        value=function((x*pre.unsqueeze(-1)).sum(-2))
        mixed=torch.einsum('...ij,...jd->...id',residual,x)
        return norm(mixed+post.unsqueeze(-1)*value.unsqueeze(-2))


class MHCStack(nn.Module):
    def __init__(self,base,streams=2,iterations=20):
        super().__init__();self.base=base;self.streams=streams
        self.attention=nn.ModuleList([HyperConnection(layer.d_model,streams,iterations) for layer in base.layers])
        self.feedforward=nn.ModuleList([HyperConnection(layer.d_model,streams,iterations) for layer in base.layers])

    def forward(self,src,mask=None,src_key_padding_mask=None,is_causal=False):
        assert mask is None and src_key_padding_mask is None and not is_causal
        x=src.unsqueeze(-2).expand(*src.shape[:-1],self.streams,src.shape[-1])
        for layer,attention,feedforward in zip(self.base.layers,self.attention,self.feedforward):
            assert not layer.norm_first
            x=attention(x,lambda value:layer._self_attn_block(value,None,None,False),layer.norm1)
            x=feedforward(x,layer._ff_block,layer.norm2)
        result=x.mean(-2)
        return self.base.norm(result) if self.base.norm is not None else result
