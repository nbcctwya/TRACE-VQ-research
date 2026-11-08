"""Mechanism checks, causal availability and frozen Stage1/restore contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from trace_vq.data.hopfield import build_candidates,neighbor_codes
from trace_vq.data.schema import TOTAL_DIM,unpack_batch
from trace_vq.models.batch67 import (VARIANTS,Batch67Predictor,GridHypergraph,
    HistoricalHopfield,SplineLayer,augment_coordinates,all_grid,grid_codes)
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.representation import FactorVQVAE


class MechanismTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(5)

    def test_spline_partition_endpoints_and_gradients(self):
        layer=SplineLayer(3,4);x=torch.linspace(-1,1,51)[:,None].expand(-1,3).clone().requires_grad_()
        basis=layer.basis(x)
        torch.testing.assert_close(basis.sum(-1),torch.ones(51,3),atol=1e-6,rtol=1e-6)
        self.assertTrue((basis>=0).all());layer(x).square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertGreater(layer.coefficients.grad.abs().sum(),0)

    def test_boundary_flip_preserves_interior_and_grid_and_rng_replays(self):
        u=torch.tensor([[0.,.49,.51],[6.51,7.,2.1]]);q=u.round()
        state=torch.get_rng_state();a=augment_coordinates(u,q,1.,.1)
        torch.set_rng_state(state);b=augment_coordinates(u,q,1.,.1)
        self.assertTrue(torch.equal(a,b));self.assertEqual(a.tolist(),[[0.,1.,0.],[6.,7.,2.]])
        self.assertTrue(torch.equal(grid_codes(a), (a.long()*torch.tensor([1,8,64])).sum(-1)))

    def test_hypergraph_is_local_permutation_equivariant_and_differentiable(self):
        layer=GridHypergraph(4);h=torch.randn(4,4,requires_grad=True)
        codes=torch.tensor([0,0,1,511]);a=layer.message(h,codes);perm=torch.tensor([3,1,0,2])
        torch.testing.assert_close(a[perm],layer.message(h[perm],codes[perm]))
        torch.testing.assert_close(a[-1],h[-1]);h2=h.detach().clone();h2[-1]+=100
        torch.testing.assert_close(a[:-1],layer.message(h2,codes)[:-1]);a.square().sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_hopfield_matches_masked_softmax_and_empty_memory_is_zero(self):
        layer=HistoricalHopfield(4,dict(key_dim=3,inverse_temperature=4))
        h=torch.randn(2,4);keys=torch.randn(2,5,4);values=torch.randn(2,5);mask=torch.tensor([[1,0,1,0,0],[0,0,0,0,0]])
        context,weights=layer.retrieve(h,keys,values,mask)
        logit=(torch.nn.functional.normalize(layer.key(keys[0,[0,2]]),dim=-1)*torch.nn.functional.normalize(layer.query(h[:1]),dim=-1)).sum(-1)*4
        w=logit.softmax(-1);expected=(w[:,None]*torch.cat((keys[0,[0,2]],values[0,[0,2],None]),-1)).sum(0)
        torch.testing.assert_close(context[0],expected);self.assertEqual(weights[1].sum(),0);self.assertEqual(context[1].abs().sum(),0)
        layer.output.bias.data.fill_(1)
        self.assertEqual(layer(h,keys,values,mask)[1].abs().sum(),0)

    def test_candidate_maturity_uses_complete_calendar_and_excludes_self(self):
        dates=np.repeat(np.array([0,1,2,6,7,9]),2);q=np.zeros((12,3),np.uint8);labels=np.arange(12,dtype='float32')
        with tempfile.TemporaryDirectory() as folder:
            spec=build_candidates(dates,q,labels,folder,days=10)
            pools=np.load(Path(folder)/'candidates.npy')
            self.assertTrue((pools[:6]==-1).all());self.assertEqual(spec['minimum_observed_lag'],6)
            for day in range(10):
                rows=pools[day][pools[day]>=0]
                self.assertTrue((dates[rows]+5<day).all())
            self.assertEqual(set(pools[6,0][pools[6,0]>=0]),{0,1})

    def test_future_labels_and_coordinates_cannot_change_earlier_memory(self):
        dates=np.repeat(np.arange(15),3);q=np.zeros((45,3),np.uint8);labels=np.arange(45,dtype='float32')
        with tempfile.TemporaryDirectory() as folder:
            build_candidates(dates,q,labels,Path(folder)/'a',days=15)
            changed=labels.copy();changed[dates>=7]=np.nan;q2=q.copy();q2[dates>=7]=7
            build_candidates(dates,q2,changed,Path(folder)/'b',days=15)
            a=np.load(Path(folder)/'a/candidates.npy');b=np.load(Path(folder)/'b/candidates.npy')
            np.testing.assert_array_equal(a[:7],b[:7])
            # Future labels alone cannot affect earlier selected IDs or values.
            build_candidates(dates,q,changed,Path(folder)/'c',days=15)
            c=np.load(Path(folder)/'c/candidates.npy');np.testing.assert_array_equal(a[:13],c[:13])
            for row in a[:13].ravel():
                if row>=0:self.assertEqual(labels[row],changed[row])

    def test_neighbor_buckets_do_not_wrap_grid_edges(self):
        self.assertEqual(set(neighbor_codes(0)),{0,1,8,64});self.assertEqual(set(neighbor_codes(511)),{511,510,503,447})


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        if not OmegaConf.has_resolver('half'):OmegaConf.register_new_resolver('half',lambda x:int(x)//2)
        c=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/config.yaml')
        c.vqvae.hidden_size=8;c.vqvae.vq_embed_dim=8;c.vqvae.decoder.hidden_channels=8
        c.predictor.moe_hidden=8;c.predictor.transformer.d_model=8;c.predictor.transformer.dim_feedforward=16
        c.vqvae.quantizer.update(research_kind='fsq',research_levels=[8,8,8]);cls.config=OmegaConf.to_container(c,resolve=True)
        torch.manual_seed(42);s1=FactorVQVAE(cls.config,T_max=2);checkpoint=cls.root/'fsq.ckpt'
        torch.save(dict(state_dict=s1.state_dict()),checkpoint);cls.config['predictor']['saved_model']=str(checkpoint)
        torch.manual_seed(11);cls.canonical=torch.randn(7,20,TOTAL_DIM)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def cfg(self,variant):
        c=copy.deepcopy(self.config);c['research']=dict(variant=variant,**VARIANTS[variant]);return c
    def batch(self,model):
        parts=unpack_batch(self.canonical)
        with torch.no_grad():h=model.encoder(model.revin(parts.stock_feature,mode='norm'));z=model.quantizer(h)[0]
        return dict(canonical=self.canonical,frozen_h=h,frozen_zq=z,memory_keys=torch.randn(7,16,8),
            memory_values=torch.randn(7,16),memory_mask=torch.ones(7,16))

    def test_all_eleven_backward_frozen_stage1_and_strict_restore_after_update(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(0);m=Batch67Predictor(self.cfg(variant),T_max=2);batch=self.batch(m)
                frozen={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('encoder.','quantizer.','revin.'))}
                m.train();m.log=lambda *a,**k:None;m.log_dict=lambda *a,**k:None
                loss=m.training_step(batch,0);loss.backward();self.assertTrue(torch.isfinite(loss))
                for name,p in m.named_parameters():
                    if name.startswith(('encoder.','quantizer.','revin.')):self.assertIsNone(p.grad)
                    elif p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
                torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4).step()
                for k,v in frozen.items():self.assertTrue(torch.equal(v,m.state_dict()[k]),k)
                m.eval();expected=m.batch_prediction(batch)[1][0]
                restored=Batch67Predictor(self.cfg(variant),T_max=2).eval();restored.load_state_dict(m.state_dict(),strict=True)
                self.assertTrue(torch.equal(expected,restored.batch_prediction(batch)[1][0]))

    def test_zero_added_branches_preserve_baseline_predictions(self):
        for variant in ('fsq_shrink','fsq_grid_smooth','fsq_code_aug','code_hypergraph','hyper_loading','hopfield_retrieval'):
            with self.subTest(variant=variant):
                cfg=self.cfg(variant);torch.manual_seed(0);baseline=ScreeningPredictor(cfg,T_max=2).eval()
                torch.manual_seed(0);m=Batch67Predictor(cfg,T_max=2).eval();batch=self.batch(m)
                self.assertTrue(torch.equal(baseline.batch_prediction(batch)[1][0],m.batch_prediction(batch)[1][0]))

    def test_grid_penalty_only_in_training_and_parent_mapping_and_unseen_fallback(self):
        m=Batch67Predictor(self.cfg('fsq_shrink'),T_max=2);batch=self.batch(m);parts,result=m.batch_prediction(batch)
        native,_=ScreeningPredictor.training_loss(m,parts,result);total,terms=m.training_loss(parts,result)
        self.assertIn('grid_regularizer',terms);self.assertTrue(torch.equal(native,total))
        expected=(all_grid()//2*torch.tensor([1,4,16])).sum(-1);self.assertTrue(torch.equal(m.parents,expected))
        self.assertEqual(m.parents.min(),0);self.assertEqual(m.parents.max(),63)
        m.eval();m.batch_prediction(batch);self.assertIsNone(m.grid_penalty)


if __name__=='__main__':unittest.main()
