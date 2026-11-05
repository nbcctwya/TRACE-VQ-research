"""Independent batch04 routing experiments; native MoE remains unchanged."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from trace_vq.models.experts.shared_routed import SimpleMLP, SparseDispatcher


class ResearchMoE(nn.Module):
    def __init__(self,base,settings):
        super().__init__()
        self.base,self.settings=base,settings
        self.mode=settings["routing"]
        self.route_ids=None
        self.stats={};self.stat_days=0
        if self.mode=="bias_balance":
            self.register_buffer("route_bias",torch.zeros(base.num_experts))
            self.register_buffer("load_ema",torch.full((base.num_experts,),1/base.num_experts))
        if self.mode=="finegrained":
            experts=settings["experts"]; width=settings["expert_width"]
            base.num_experts,base.k=experts,settings["active_experts"]
            base.experts=nn.ModuleList([SimpleMLP(base.expert_input_size,width,base.hidden_size) for _ in range(experts)])
            base.gate[-1]=nn.Linear(base.hidden_size,experts)
            base.noise[-1]=nn.Linear(base.hidden_size,experts)
            base.W_h=nn.Parameter(torch.eye(experts))
        if self.mode=="fsq_sort":
            base.experts.append(SimpleMLP(base.expert_input_size,base.expert_input_size,base.hidden_size))
            base.num_experts,base.k=3,1
        if self.mode=="soft_slots":
            self.slot_queries=nn.Parameter(torch.randn(base.num_experts,settings["slots_per_expert"],base.gate_input_size)/math.sqrt(base.gate_input_size))
        if self.mode in ("soft_slots","fsq_sort"):
            base.gate.requires_grad_(False);base.noise.requires_grad_(False);base.W_h.requires_grad_(False)

    def original_scores(self,z):
        clean=self.base.gate(z)
        if self.base.noisy_gating and self.training:
            deviation=self.base.softplus(self.base.noise(z))+1e-2
            logits=(clean+torch.randn_like(clean)*deviation)@self.base.W_h
        else:
            logits=clean
        return logits.softmax(-1)

    def gates(self,z):
        b=self.base
        if self.mode in ("finegrained","all_experts"):
            return b.noisy_top_k_gating(z,self.training)
        if self.mode=="fsq_sort":
            if self.route_ids is None: raise ValueError("Aligned FSQ route IDs required")
            gates=F.one_hot(self.route_ids.long(),3).to(z.dtype)
            return gates,gates.sum(0)
        probabilities=self.original_scores(z)
        if self.mode=="expert_choice":
            capacity=max(1,math.ceil(len(z)*self.settings.get("capacity_factor",1)*b.k/b.num_experts))
            chosen=probabilities.topk(min(capacity,len(z)),dim=0).indices
            mask=torch.zeros_like(probabilities).scatter(0,chosen,1)
            unnormalized=probabilities*mask
            gates=unnormalized/(unnormalized.sum(-1,keepdim=True)+1e-6)
        elif self.mode=="bias_balance":
            chosen=(probabilities+self.route_bias.detach()).topk(b.k,dim=-1).indices
            selected=probabilities.gather(-1,chosen)
            gates=torch.zeros_like(probabilities).scatter(-1,chosen,selected/(selected.sum(-1,keepdim=True)+1e-6))
            if self.training:
                with torch.no_grad():
                    load=(gates>0).sum(0).float()/(len(z)*b.k)
                    self.load_ema.mul_(self.settings["load_ema_decay"]).add_(load,alpha=1-self.settings["load_ema_decay"])
                    self.route_bias.add_(torch.sign(1/b.num_experts-self.load_ema),alpha=self.settings["bias_step"])
                    self.route_bias.sub_(self.route_bias.mean())
        else:
            raise ValueError(self.mode)
        return gates,(gates>0).sum(0)

    def note(self,gates):
        counts=(gates>0).sum(-1).float()
        values=dict(selected_experts_mean=counts.mean(),selected_experts_min=counts.min(),
                    selected_experts_max=counts.max(),zero_route_fraction=(counts==0).float().mean())
        load=(gates>0).sum(0).float()
        values["load_cv_squared"]=self.base.cv_squared(load)
        for key,value in values.items():
            self.stats[key]=self.stats.get(key,0)+value.detach()
        self.stat_days+=1

    def reset_diagnostics(self):
        self.stats={};self.stat_days=0

    def diagnostics(self):
        values={k:float(v.cpu())/max(self.stat_days,1) for k,v in self.stats.items()}
        return dict(mode=self.mode,days=self.stat_days,**values)

    def forward(self,x,z,loss_coef=1):
        b=self.base
        if self.mode=="soft_slots":
            logits=torch.einsum("nd,esd->nes",z,self.slot_queries)
            dispatch=logits.softmax(dim=0)
            slots=torch.einsum("nes,nd->esd",dispatch,x)
            outputs=torch.stack([expert(slots[e]) for e,expert in enumerate(b.experts)])
            combine=logits.flatten(1).softmax(-1).reshape_as(logits)
            result=torch.einsum("nes,esd->nd",combine,outputs)
            gates=combine.sum(-1)
            auxiliary=z.sum()*0
        else:
            gates,load=self.gates(z)
            dispatcher=SparseDispatcher(b.num_experts,gates)
            inputs=dispatcher.dispatch(x)
            outputs=[expert(values) for expert,values in zip(b.experts,inputs)]
            result=dispatcher.combine(outputs)
            auxiliary=(z.sum()*0 if self.mode=="bias_balance" else
                       (b.cv_squared(gates.sum(0))+b.cv_squared(load))*loss_coef)
        if b.shared_expert is not None: result=result+b.shared_expert(x)
        self.note(gates)
        return result,auxiliary

    def parameter_counts(self):
        b=self.base
        shared=sum(p.numel() for p in b.shared_expert.parameters()) if b.shared_expert is not None else 0
        expert=sum(p.numel() for p in b.experts[0].parameters())
        routed=sum(p.numel() for module in b.experts for p in module.parameters())
        total=sum(p.numel() for p in self.parameters() if p.requires_grad)
        activated=b.num_experts if self.mode=="soft_slots" else b.k
        return dict(moe_trainable_parameters=total,moe_activated_parameters_estimate=total-routed+activated*expert,
                    shared_expert_parameters=shared,routed_expert_parameters_each=expert,routed_experts=b.num_experts,
                    activated_routed_experts=activated)
