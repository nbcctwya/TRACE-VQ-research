"""Quantizer mechanisms, causal pairs and exact two-stage restore contracts."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.batch89 import PairedCanonicalDays
from trace_vq.data.schema import TOTAL_DIM,unpack_batch
from trace_vq.models.batch89 import VARIANTS,Batch89Stage1,Batch89Predictor,symmetric_contrastive
from trace_vq.models.batch89_quantizers import (ResearchFSQ,GroupedFSQ,ResidualFSQ,
    BinarySphericalQuantizer,SinkhornQuantizer,rank_bins,sinkhorn,cross_correlation)
from trace_vq.models.quantization.research import FiniteScalarQuantiser


class MechanismTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(8)

    def test_rank_quantiles_preserve_ties_and_stock_permutations(self):
        raw=torch.arange(16.)[:,None].expand(-1,3).clone();levels=torch.tensor([8,8,8])
        q=rank_bins(raw,levels);self.assertEqual(q[:,0].tolist(),list(np.repeat(np.arange(8),2)))
        permutation=torch.randperm(16);self.assertTrue(torch.equal(q[permutation],rank_bins(raw[permutation],levels)))
        self.assertTrue(torch.equal(rank_bins(torch.ones_like(raw),levels),torch.full_like(raw,4)))

    def test_cross_section_standardization_is_shift_invariant_and_has_gradient(self):
        base=FiniteScalarQuantiser([8,8,8],3);base.to_grid.weight.data.copy_(torch.eye(3));base.to_grid.bias.data.zero_()
        q=ResearchFSQ(base,dict(quantizer='cs'));h=torch.randn(30,3,requires_grad=True)
        a=q(h);b=q(h+torch.tensor([8.,-9.,4.]))
        self.assertTrue(torch.equal(a[2][2],b[2][2]));a[0].square().mean().backward()
        self.assertTrue(torch.isfinite(h.grad).all());self.assertGreater(h.grad.abs().sum(),0)

    def test_scale_one_matches_native_and_dither_rng_replays_and_eval_is_hard(self):
        base=FiniteScalarQuantiser([8,8,8],8);h=torch.randn(25,8)
        scaled=ResearchFSQ(base,dict(quantizer='scale'));torch.testing.assert_close(scaled(h)[0],base(h)[0])
        noisy=ResearchFSQ(base,dict(quantizer='dither')).train();state=torch.get_rng_state();a=noisy(h)[0]
        torch.set_rng_state(state);self.assertTrue(torch.equal(a,noisy(h)[0]))
        noisy.eval();self.assertTrue(torch.equal(noisy(h)[0],base(h)[0]))

    def test_log_sinkhorn_matches_direct_scaling_and_balances_soft_mass(self):
        logits=torch.randn(19,11,requires_grad=True)*.5;q=sinkhorn(logits,30)
        reference=logits.exp()
        for _ in range(30):
            reference=reference/reference.sum(0,keepdim=True)*(19/11)
            reference=reference/reference.sum(1,keepdim=True)
        torch.testing.assert_close(q,reference,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(q.sum(0),torch.full((11,),19/11),atol=1e-5,rtol=1e-5)
        model=SinkhornQuantizer(8,dict(sinkhorn_epsilon=.1,sinkhorn_iterations=20,commit_weight=.25))
        h=torch.randn(7,8,requires_grad=True);z,loss,info=model(h)
        torch.testing.assert_close(z,model.codebook[info[2]],atol=1e-7,rtol=1e-6)
        (z.square().mean()+loss).backward();self.assertGreater(h.grad.abs().sum(),0);self.assertGreater(model.codebook.grad.abs().sum(),0)
        self.assertLessEqual(model.last_diagnostics['hard_used'],7)

    def test_bsq_is_spherical_binary_with_joint_entropy_not_binary_fsq(self):
        native=FiniteScalarQuantiser([2]*9,8);settings=dict(bsq_temperature=.1,bsq_entropy_weight=.1,bsq_commit_weight=0.)
        model=BinarySphericalQuantizer(native,settings);h=torch.randn(13,8,requires_grad=True)
        sphere=model.spherical(h);torch.testing.assert_close(sphere.norm(dim=-1),torch.ones(13))
        self.assertEqual(model.vertices.shape,(512,9));torch.testing.assert_close(model.vertices.norm(dim=-1),torch.ones(512))
        z,loss,info=model(h);p=(2*sphere@model.vertices.T/.1).softmax(-1);marginal=p.mean(0)
        expected=.1*(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()+(marginal*marginal.clamp_min(1e-12).log()).sum())
        torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-6)
        self.assertTrue(((info[2]>=0)&(info[2]<512)).all());(z.square().mean()+loss).backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_product_quantizers_never_allocate_joint_one_hot(self):
        for model,limit in ((GroupedFSQ(8),512**2),(ResidualFSQ(8),27*512)):
            h=torch.randn(19,8,requires_grad=True);z,loss,info=model(h)
            self.assertEqual(z.shape,h.shape);self.assertEqual(info[1].shape,(19,0))
            self.assertTrue(((info[2]>=0)&(info[2]<limit)).all());z.square().mean().backward()
            self.assertTrue(torch.isfinite(h.grad).all())

    def test_correlation_regularizers_handle_constant_styles_and_have_gradient(self):
        x=torch.randn(25,3,requires_grad=True);style=torch.randn(25,4);style[:,0]=1
        corr=cross_correlation(x,style);self.assertEqual(corr[:,0].abs().sum(),0)
        corr.square().mean().backward();self.assertGreater(x.grad.abs().sum(),0);self.assertTrue(torch.isfinite(x.grad).all())

    def test_contrast_uses_same_instrument_targets_and_permutation_invariance(self):
        a=torch.eye(5,requires_grad=True);good=symmetric_contrastive(a,a,.1)
        bad=symmetric_contrastive(a,a.roll(1,0),.1);self.assertLess(good,bad)
        p=torch.tensor([4,2,3,0,1]);torch.testing.assert_close(good,symmetric_contrastive(a[p],a[p],.1))
        good.backward();self.assertTrue(torch.isfinite(a.grad).all())


class PairTests(unittest.TestCase):
    def test_missing_market_day_is_not_replaced_by_previous_observed_day(self):
        calendar=pd.date_range('2020-01-01',periods=5)
        index=pd.MultiIndex.from_tuples([(calendar[d],s) for d in (0,1,3,4) for s in ('A','B') if not(d==4 and s=='B')],names=['datetime','instrument'])
        class Source:
            market_frame=pd.DataFrame(index=calendar)
            def __len__(self):return len(index)
            def get_index(self):return index
            def __getitem__(self,positions):return np.broadcast_to(np.asarray(positions)[...,None,None],(*np.asarray(positions).shape,20,TOTAL_DIM)).copy()
        pairs=PairedCanonicalDays(Source());day3=pairs[pairs.days[2]];self.assertEqual(len(day3['align_current']),0)
        last=pairs[pairs.days[3]];self.assertEqual(last['previous_canonical'].shape[0],2)
        self.assertEqual(last['align_current'].tolist(),[0]);self.assertEqual(last['align_previous'].tolist(),[0])


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        if not OmegaConf.has_resolver('half'):OmegaConf.register_new_resolver('half',lambda x:int(x)//2)
        c=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/config.yaml')
        c.vqvae.hidden_size=8;c.vqvae.vq_embed_dim=8;c.vqvae.decoder.hidden_channels=8
        c.predictor.moe_hidden=8;c.predictor.transformer.d_model=8;c.predictor.transformer.dim_feedforward=16
        cls.config=OmegaConf.to_container(c,resolve=True);torch.manual_seed(11);cls.canonical=torch.randn(7,20,TOTAL_DIM)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def cfg(self,variant):
        c=copy.deepcopy(self.config);c['research']=dict(variant=variant,**VARIANTS[variant])
        c['vqvae']['quantizer']['research_kind']='batch89';c['vqvae']['predictor']['pred_weight']=VARIANTS[variant].get('pred_weight',.0001)
        c['artifact_root']=str(self.root);return c

    def test_all_nineteen_stage1_and_stage2_backward_freeze_and_strict_restore(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config=self.cfg(variant);torch.manual_seed(42);model=Batch89Stage1(config,T_max=2).train()
                model.log=lambda *a,**k:None;batch=self.canonical
                if variant=='fsq_contrast':batch=dict(canonical=batch,previous_canonical=batch.flip(0),align_current=torch.arange(7),align_previous=torch.arange(6,-1,-1))
                loss=model.training_step(batch,0);loss.backward();self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                torch.optim.AdamW(model.parameters(),lr=1e-4).step();model.eval()
                restored=Batch89Stage1(config,T_max=2).eval();restored.load_state_dict(model.state_dict(),strict=True)
                parts=unpack_batch(self.canonical)
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(parts.stock_feature,parts.prior_factor,parts.future_returns)[4],restored(parts.stock_feature,parts.prior_factor,parts.future_returns)[4]))
                path=self.root/(variant+'.ckpt');torch.save(dict(state_dict=model.state_dict()),path)
                config['predictor']['saved_model']=str(path);torch.manual_seed(0);p=Batch89Predictor(config,T_max=2).eval()
                with torch.no_grad():h=p.encoder(p.revin(parts.stock_feature,mode='norm'));z=p.quantizer(h)[0]
                frozen=dict(canonical=self.canonical,frozen_h=h,frozen_zq=z)
                before={k:v.clone() for k,v in p.state_dict().items() if k.startswith(('encoder.','quantizer.','revin.'))}
                p.train();p.log=lambda *a,**k:None;p.log_dict=lambda *a,**k:None
                objective=p.training_step(frozen,0);objective.backward();self.assertTrue(torch.isfinite(objective))
                torch.optim.AdamW([v for v in p.parameters() if v.requires_grad],lr=1e-4).step()
                for name,value in before.items():self.assertTrue(torch.equal(value,p.state_dict()[name]),name)
                p.eval();expected=p.batch_prediction(frozen)[1][0]
                again=Batch89Predictor(config,T_max=2).eval();again.load_state_dict(p.state_dict(),strict=True)
                self.assertTrue(torch.equal(expected,again.batch_prediction(frozen)[1][0]))


if __name__=='__main__':unittest.main()
