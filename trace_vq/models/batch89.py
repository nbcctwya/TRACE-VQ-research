"""Independent Stage1+Stage2 experiments for checklist 39–53 only."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trace_vq.cli.next_screen import QuietFactorVQVAE
from trace_vq.data.frozen import atomic_json
from trace_vq.data.schema import unpack_batch
from trace_vq.models.batch89_quantizers import build_research_quantizer,cross_correlation
from trace_vq.models.screening import ScreeningPredictor


BATCH08={
    's1_predw_1e3':dict(item='F1',quantizer='native',pred_weight=.001),
    's1_predw_1e2':dict(item='F1',quantizer='native',pred_weight=.01),
    's1_predw_1e1':dict(item='F1',quantizer='native',pred_weight=.1),
    'fsq_ifsq':dict(item='A1',quantizer='ifsq',bound_scale=.8),
    'fsq_cs':dict(item='A2',quantizer='cs',cross_section_eps=1e-5),
    'fsq_rank':dict(item='A3',quantizer='rank',ties='average_rank',ste='bounded_native_projection'),
    'fsq_levels_44442':dict(item='A4',quantizer='native',levels=[4,4,4,4,2]),
    'fsq_levels_binary9':dict(item='A4',quantizer='native',levels=[2]*9),
    'fsq_levels_16162':dict(item='A4',quantizer='native',levels=[16,16,2]),
    'sinkhorn_quant':dict(item='J8',quantizer='sinkhorn',sinkhorn_epsilon=.1,sinkhorn_iterations=20,commit_weight=.25,
        ste='soft_assignment_reconstruction',assignment_scope='full_current_cross_section_train_and_eval'),
    'bsq_tokenizer':dict(item='K7',quantizer='bsq',levels=[2]*9,bsq_temperature=.1,bsq_entropy_weight=.1,
        bsq_commit_weight=0.,sphere_dimension=9,entropy='exact_joint_512_codes'),
}
BATCH09={
    'fsq_res':dict(item='A5',quantizer='residual',coarse_levels=[3,3,3],fine_levels=[8,8,8],residual_normalization='LayerNorm'),
    'fsq_scale':dict(item='A6',quantizer='scale',initial_scale=1.,log_scale_bounds=[-5,5]),
    'fsq_dither':dict(item='A7',quantizer='dither',uniform_noise=[-.5,.5],training_only=True,endpoints='clamp'),
    'fsq_ortho':dict(item='A8',quantizer='ortho',orthogonal_weight=.01),
    'fsq_style_orth':dict(item='A9',quantizer='style',style_weight=.01,style_names=['ROC20','STD20','BETA20','VSTD20'],style_indices=[15,25,30,135]),
    'fsq_hybrid':dict(item='A10',quantizer='hybrid',continuous_dim=4,continuous_activation='tanh',joint_projection_dimension=128),
    'fsq_group':dict(item='A11',quantizer='grouped',groups=2,levels_per_group=[8,8,8],joint_one_hot=False),
    'fsq_contrast':dict(item='A13',quantizer='native',contrast_weight=.1,contrast_temperature=.1,
        pairing='same_instrument_previous_market_day',negative_pool='other_instruments_same_date',symmetric=True),
}
VARIANTS={**{v:dict(batch=8,family='fsq',**s) for v,s in BATCH08.items()},
          **{v:dict(batch=9,family='fsq',**s) for v,s in BATCH09.items()}}


def construction_config(config):
    c=deepcopy(config);q=c['vqvae']['quantizer'];q['research_kind']='fsq'
    q['research_levels']=config['research'].get('levels',[8,8,8]);c['vqvae']['num_embed']=512
    return c


def replace_quantizer(original,settings):
    state=torch.get_rng_state()
    try:return build_research_quantizer(original,settings)
    finally:torch.set_rng_state(state)


def symmetric_contrastive(current,previous,temperature):
    if len(current)<2:return current.sum()*0
    a,b=F.normalize(current,dim=-1),F.normalize(previous,dim=-1)
    logits=a@b.T/temperature;target=torch.arange(len(a),device=a.device)
    return (F.cross_entropy(logits,target)+F.cross_entropy(logits.T,target))/2


class Batch89Stage1(QuietFactorVQVAE):
    def __init__(self,config,T_max):
        original=deepcopy(config);super().__init__(construction_config(config),T_max)
        self.config=original;self.vqvae.config=original;self.research=original['research']
        self.vqvae.quantizer=replace_quantizer(self.vqvae.quantizer,self.research)
        self.current_h=None
        self.vqvae.spatial_encoder.register_forward_hook(self.capture_h)

    def capture_h(self,module,inputs,output):self.current_h=output

    def training_step(self,batch,batch_idx):
        canonical=batch['canonical'] if isinstance(batch,dict) else batch
        parts=unpack_batch(canonical.float());values=self(parts.stock_feature,parts.prior_factor,parts.future_returns)
        recon,quant,pred,total,z,info=values;extra=total*0;q=self.vqvae.quantizer
        kind=self.research['quantizer']
        if kind in ('ortho','style'):
            raw=q.last_raw
            if kind=='ortho':
                corr=cross_correlation(raw,raw);extra=self.research['orthogonal_weight']*(corr-torch.diag_embed(corr.diagonal())).square().sum()/(raw.shape[-1]*(raw.shape[-1]-1))
            else:
                styles=parts.stock_feature[:,-1,self.research['style_indices']].detach()
                extra=self.research['style_weight']*cross_correlation(raw,styles).square().mean()
        if self.research.get('contrast_weight') and batch['align_current'].numel()>1:
            # Read full previous-day cross-section so the native cross-stock
            # encoder never receives a reduced, stock-matched subset.
            previous=unpack_batch(batch['previous_canonical'].float())
            a=self.current_h
            b=self.vqvae.spatial_encoder(self.vqvae.revin(previous.stock_feature,mode='norm'))
            extra=self.research['contrast_weight']*symmetric_contrastive(a[batch['align_current'].long()],b[batch['align_previous'].long()],self.research['contrast_temperature'])
        loss=total+extra
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite Stage1 loss')
        self.epoch_vq_codes.append(info[2].detach().cpu().numpy())
        for name,value in dict(loss=loss,recon_loss=recon,vq_loss=quant,pred_loss=pred,research_loss=extra).items():
            self.log('train_'+name,value,on_step=False,on_epoch=True,batch_size=len(z))
        return loss

    def validation_step(self,batch,batch_idx):
        canonical=batch['canonical'] if isinstance(batch,dict) else batch
        parts=unpack_batch(canonical.float());recon,quant,pred,total,_,_=self(parts.stock_feature,parts.prior_factor,parts.future_returns)
        # Additional correlation/temporal penalties are training-only. Native
        # quantizer objectives and the declared prediction weight remain here.
        for name,value in dict(loss=total,recon_loss=recon,vq_loss=quant,pred_loss=pred).items():
            self.log('val_'+name,value,on_step=False,on_epoch=True,batch_size=len(canonical))
        return total

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        diagnostic=getattr(self.vqvae.quantizer,'last_diagnostics',{})
        if diagnostic:atomic_json(Path(self.config['artifact_root'])/f'quantizer_epoch{self.current_epoch:03d}.json',diagnostic)


class Batch89Predictor(ScreeningPredictor):
    def __init__(self,config,T_max):
        original=deepcopy(config);super().__init__(construction_config(config),T_max)
        self.config=original

    def load_pretrained_vqvae(self,checkpoint_path=None):
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)['state_dict']
        self.quantizer=replace_quantizer(self.quantizer,self.config['research'])
        for destination,prefix in ((self.encoder,'vqvae.spatial_encoder.'),(self.quantizer,'vqvae.quantizer.'),(self.revin,'vqvae.revin.')):
            destination.load_state_dict({k[len(prefix):]:v for k,v in checkpoint.items() if k.startswith(prefix)},strict=True)
