"""Mechanism, leakage and latest-epoch recovery tests for batch05."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader

from tests.test_batch02 import TinyDays,TinyPredictor
from trace_vq.cli.screening import initialize
from trace_vq.data.frozen import DailyPositions
from trace_vq.data.schema import TOTAL_DIM,unpack_batch
from trace_vq.models.batch05 import VARIANTS,Batch05Predictor
from trace_vq.models.prediction.batch05_temporal import recurrent_kda,TorchKDA,HyperConnection
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.representation import FactorVQVAE
from trace_vq.training.resilient import AtomicEpochCheckpoint,BatchHeartbeat
from trace_vq.training.research_run import capture_rng


class MechanismTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(19)

    def test_kda_matches_explicit_transition_matrix_outputs_and_gradients(self):
        q,k,v=torch.randn(2,6,2,4),torch.randn(2,6,2,4)*.2,torch.randn(2,6,2,3)
        g,beta=-torch.rand(2,6,2,4),torch.rand(2,6,2)
        inputs=[x.requires_grad_() for x in (q,k,v,g,beta)]
        actual,state=recurrent_kda(*inputs);S=q.new_zeros(2,2,4,3);outputs=[]
        for t in range(6):
            key=k[:,t];b=beta[:,t,:,None,None]
            transition=(torch.eye(4)-b*key.unsqueeze(-1)*key.unsqueeze(-2))@torch.diag_embed(g[:,t].exp())
            S=transition@S+b*key.unsqueeze(-1)*v[:,t].unsqueeze(-2)
            outputs.append((q[:,t].unsqueeze(-2)@S).squeeze(-2)/2)
        expected=torch.stack(outputs,1)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(state,S,rtol=1e-5,atol=1e-6)
        a=torch.autograd.grad(actual.square().sum()+state.square().sum(),inputs,retain_graph=True)
        b=torch.autograd.grad(expected.square().sum()+S.square().sum(),inputs)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=2e-5,atol=2e-6)

    def test_kda_is_causal_and_readout_depends_on_past_tokens(self):
        model=TorchKDA(8,2).eval();x=torch.randn(3,9,8,requires_grad=True)
        y=model(x);changed=x.detach().clone();changed[:,5:]+=10
        self.assertTrue(torch.equal(y[:,:5],model(changed)[:,:5]))
        gradient=torch.autograd.grad(y[:,-1].square().sum(),x)[0]
        self.assertGreater(float(gradient[:,:-1].abs().sum()),0)

    def test_mhc_has_nonnegative_doubly_stochastic_residual_and_gradients(self):
        model=HyperConnection(8,2,20);x=torch.randn(3,5,2,8,requires_grad=True)
        pre,post,res=model.coefficients(x)
        torch.testing.assert_close(res.sum(-1),torch.ones_like(res.sum(-1)),atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(res.sum(-2),torch.ones_like(res.sum(-2)),atol=1e-6,rtol=1e-6)
        self.assertTrue((res>0).all() and (pre>0).all() and (post>0).all())
        (pre.square().sum()+post.square().sum()+res.square().sum()).backward()
        self.assertTrue(torch.isfinite(model.phi.weight.grad).all())
        self.assertGreater(float(model.phi.weight.grad.abs().sum()),0)


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        if not OmegaConf.has_resolver('half'):OmegaConf.register_new_resolver('half',lambda x:int(x)//2)
        c=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/config.yaml')
        c.vqvae.hidden_size=8;c.vqvae.vq_embed_dim=8;c.vqvae.decoder.hidden_channels=8
        c.predictor.moe_hidden=8;c.predictor.transformer.d_model=8;c.predictor.transformer.dim_feedforward=16
        cls.config=OmegaConf.to_container(c,resolve=True);cls.checkpoints={}
        for family in ('vq','fsq'):
            cfg=copy.deepcopy(cls.config)
            if family=='fsq':cfg['vqvae']['quantizer'].update(research_kind='fsq',research_levels=[8,8,8])
            torch.manual_seed(42);s1=FactorVQVAE(cfg,T_max=2);path=cls.root/(family+'.ckpt')
            torch.save(dict(state_dict=s1.state_dict()),path);cls.checkpoints[family]=str(path)
        torch.manual_seed(11);cls.canonical=torch.randn(7,20,TOTAL_DIM)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def cfg(self,variant):
        c=copy.deepcopy(self.config);settings=VARIANTS[variant]
        c['research']=dict(variant=variant,batch=5,**settings)
        c['predictor']['saved_model']=self.checkpoints[settings['family']]
        if settings['family']=='fsq':c['vqvae']['quantizer'].update(research_kind='fsq',research_levels=[8,8,8])
        if variant=='raven_context_experts':c['predictor']['n_expert']=3
        return c

    def batch(self,model):
        parts=unpack_batch(self.canonical)
        with torch.no_grad():h=model.encoder(model.revin(parts.stock_feature,mode='norm'));z=model.quantizer(h)[0]
        return dict(canonical=self.canonical,frozen_h=h,frozen_zq=z)

    def test_all_four_predictors_backward_frozen_and_strict_restore(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(0);model=Batch05Predictor(self.cfg(variant),T_max=2);batch=self.batch(model)
                model.log=lambda *a,**kw:None;model.log_dict=lambda *a,**kw:None
                loss=model.training_step(batch,0);loss.backward();self.assertTrue(torch.isfinite(loss))
                for name,p in model.named_parameters():
                    if name.startswith(('encoder.','quantizer.','revin.')):self.assertIsNone(p.grad)
                    elif p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
                model.eval();expected=model.batch_prediction(batch)[1][0]
                restored=Batch05Predictor(self.cfg(variant),T_max=2).eval();restored.load_state_dict(model.state_dict(),strict=True)
                self.assertTrue(torch.equal(expected,restored.batch_prediction(batch)[1][0]))

    def test_cross_attention_zero_branch_starts_exactly_at_native_baseline(self):
        cfg=self.cfg('s2_cross_attn');torch.manual_seed(0);base=ScreeningPredictor(cfg,T_max=2).eval()
        torch.manual_seed(0);model=Batch05Predictor(cfg,T_max=2).eval();batch=self.batch(model)
        self.assertTrue(torch.equal(base.batch_prediction(batch)[1][0],model.batch_prediction(batch)[1][0]))

    def test_cross_attention_is_stock_permutation_equivariant(self):
        model=Batch05Predictor(self.cfg('s2_cross_attn'),T_max=2).eval()
        torch.nn.init.normal_(model.loadings.cross.out_proj.weight,std=.1)
        batch=self.batch(model);permutation=torch.tensor([2,0,6,4,1,5,3])
        expected=model.batch_prediction(batch)[1][0][permutation]
        actual=model.batch_prediction({k:v[permutation] for k,v in batch.items()})[1][0]
        torch.testing.assert_close(expected,actual,atol=2e-5,rtol=2e-5)

    def test_short_window_towers_cannot_read_earlier_features(self):
        model=Batch05Predictor(self.cfg('raven_context_experts'),T_max=2).eval()
        feature=unpack_batch(self.canonical).stock_feature
        for tower in model.loadings.towers:
            changed=feature.clone();changed[:,:-tower.window]+=100
            self.assertTrue(torch.equal(tower(feature),tower(changed)))


class PlateauPredictor(TinyPredictor):
    def validation_step(self,batch,batch_idx):self.log('val_loss',1.,on_epoch=True,batch_size=len(batch))


class StopAfterTwo(Callback):
    def on_train_epoch_end(self,trainer,pl_module):trainer.should_stop=trainer.current_epoch==1


class RecoveryTests(unittest.TestCase):
    def test_latest_checkpoint_advances_without_improvement_and_resume_is_exact(self):
        torch.set_num_threads(1)
        def run(folder,device,stop=False,resume=False):
            folder.mkdir(parents=True,exist_ok=True);initialize(17);model=PlateauPredictor();data=TinyDays()
            train=DataLoader(data,sampler=DailyPositions(data,True),batch_size=None,num_workers=2)
            valid=DataLoader(data,sampler=DailyPositions(data),batch_size=None,num_workers=2)
            record=BatchHeartbeat(folder,device);best=AtomicEpochCheckpoint(dirpath=folder/'checkpoints',monitor='val_loss',
                save_top_k=1,save_last=True,save_on_train_epoch_end=True,enable_version_counter=False)
            fit=pl.Trainer(max_epochs=4,accelerator='gpu' if device=='cuda' else 'cpu',devices=1,logger=False,
                callbacks=[record]+([StopAfterTwo()] if stop else [])+[best],enable_progress_bar=False,enable_model_summary=False,deterministic=True)
            fit.fit(model,train_dataloaders=train,val_dataloaders=valid,ckpt_path=str(folder/'checkpoints/last.ckpt') if resume else None)
            c=torch.load(folder/'checkpoints/last.ckpt',map_location='cpu',weights_only=True)
            self.assertEqual(c['epoch'],1 if stop else 3)
            self.assertEqual(torch.load(best.best_model_path,map_location='cpu',weights_only=True)['epoch'],0)
            return copy.deepcopy(model.state_dict()),capture_rng(device),record.rows
        with tempfile.TemporaryDirectory() as directory:
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                root=Path(directory)/device;expected,rng,rows=run(root/'full',device);run(root/'split',device,True)
                actual,rng2,rows2=run(root/'split',device,resume=True)
                for key in expected:self.assertTrue(torch.equal(expected[key],actual[key]),key)
                self.assertTrue(torch.equal(rng['torch'],rng2['torch']))
                self.assertEqual(rng['python'],rng2['python']);self.assertEqual(rng['numpy'],rng2['numpy'])
                if device=='cuda':self.assertTrue(all(torch.equal(a,b) for a,b in zip(rng['cuda'],rng2['cuda'])))
                self.assertEqual([r['val_loss'] for r in rows],[r['val_loss'] for r in rows2])


if __name__=='__main__':unittest.main()
