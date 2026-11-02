"""Causal context, meaningful ranking gradients, and research quantizer invariants."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.frozen import DailyPositions
from trace_vq.data.research_context import PairedDays, build_history_positions, CodeHistory
from trace_vq.data.schema import TOTAL_DIM, unpack_batch
from trace_vq.models.next_screen import NextPredictor, NEXT_VARIANTS, ranknet_objective, standardized_turnover
from trace_vq.models.quantization.codebook import VectorQuantiser
from trace_vq.models.quantization.research import build_quantizer, ResidualVectorQuantiser, FiniteScalarQuantiser
from trace_vq.models.representation import VQVAE
from trace_vq.training.representation import FactorVQVAE
from trace_vq.models.screening import ScreeningPredictor

ROOT = Path(__file__).resolve().parents[1]


class CachedToy:
    def __init__(self, index, offset=0):
        self.index = index
        self.offset = offset
    def __len__(self):
        return len(self.index)
    def get_index(self):
        return self.index
    def __getitem__(self, positions):
        p = np.asarray(positions)
        values = np.broadcast_to((p + self.offset)[..., None], (*p.shape, 4)).astype('float32').copy()
        return {'canonical': values, 'frozen_h': values, 'frozen_zq': values}


class NextContextTests(unittest.TestCase):
    def test_pair_alignment_is_previous_date_same_instrument(self):
        idx = pd.MultiIndex.from_tuples([(pd.Timestamp(date), name) for date, name in
              [('2020-01-02','B'), ('2020-01-01','A'), ('2020-01-02','A'),
               ('2020-01-01','B'), ('2020-01-03','C'), ('2020-01-03','B')]],
               names=['datetime','instrument'])
        source = CachedToy(idx)
        paired = PairedDays(source)
        for number, rows in enumerate(DailyPositions(source).positions):
            item = paired[rows]
            if number == 0:
                self.assertEqual(len(item['align_current']), 0)
            else:
                current = idx.take(rows[item['align_current']])
                previous_rows = item['prev_canonical'][:, 0].astype(int)[item['align_previous']]
                previous = idx.take(previous_rows)
                self.assertTrue((current.get_level_values('datetime') > previous.get_level_values('datetime')).all())
                self.assertEqual(list(current.get_level_values('instrument')), list(previous.get_level_values('instrument')))

    def test_history_is_past_only_missing_masked_and_crosses_split_boundary(self):
        indices = [pd.MultiIndex.from_tuples([(pd.Timestamp(d), s) for d,s in values],
                    names=['datetime','instrument']) for values in [
                    [('2020-01-02','B'),('2020-01-01','A'),('2020-01-02','A')],
                    [('2020-01-03','A'),('2020-01-03','B')],
                    [('2020-01-06','B'),('2020-01-06','C')]]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [dict(source_sha256=str(i), stage1_sha256='fixed') for i in range(3)]
            metadata = build_history_positions(indices, root/'history/csi300', identities, days=2)
            full = indices[0].append(indices[1:])
            positions = np.load(root/'history/csi300/positions.npy')
            for row in range(len(full)):
                for past in positions[row]:
                    if past >= 0:
                        self.assertLess(full[past][0], full[row][0])
                        self.assertEqual(full[past][1], full[row][1])
            self.assertTrue((positions[-1] == -1).all())  # New stock C has no invented history.
            for i, name in enumerate(('train','valid','test')):
                folder=root/'cache/csi300'; folder.mkdir(parents=True, exist_ok=True)
                values = np.zeros((len(indices[i]),2,4),dtype='float32')
                values[:,1] = np.arange(metadata['offsets'][i],metadata['offsets'][i+1])[:,None]
                np.save(folder/f'{name}.npy',values)
                (folder/f'{name}.json').write_text(json.dumps({'source_sha256':str(i), 'identity':{'stage1_sha256':'fixed'}}))
            wrapped = CodeHistory(CachedToy(indices[1],3),root/'history',root/'cache','csi300','valid')
            item=wrapped[np.arange(2)]
            self.assertGreater(item['history_mask'].sum(),0)
            self.assertEqual(item['code_history'][0,-1,0],2)  # A on preceding train date.


class NextLossTests(unittest.TestCase):
    def test_ranknet_penalizes_collapsed_scores_and_has_correct_gradients(self):
        prediction=torch.zeros(3,requires_grad=True)
        target=torch.tensor([1.,0.,-1.])
        loss=ranknet_objective(prediction,target)
        self.assertGreater(float(loss),0)
        loss.backward()
        self.assertLess(prediction.grad[0],0)
        self.assertGreater(prediction.grad[-1],0)
        good=ranknet_objective(target,target)
        bad=ranknet_objective(-target,target)
        self.assertLess(good,bad)

    def test_ranknet_nan_labels_and_top_weighting_are_finite(self):
        p=torch.tensor([.1,.2,.7,.5],requires_grad=True)
        y=torch.tensor([1.,-1.,0.,float('nan')])
        value=ranknet_objective(p,y,topk=1,boost=2)
        value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[-1]),0)
        self.assertEqual(float(ranknet_objective(p,torch.ones(4))),0)

    def test_turnover_uses_cross_section_normalization(self):
        x=torch.tensor([-.9,.1,1.3])
        self.assertAlmostEqual(float(standardized_turnover(x,x+5)),0,places=6)
        self.assertGreater(float(standardized_turnover(x,-x)),0)


class NextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        if not OmegaConf.has_resolver('half'):
            OmegaConf.register_new_resolver('half',lambda x:int(x)//2)
        cfg=OmegaConf.load(ROOT/'configs/config.yaml')
        cfg.vqvae.hidden_size=8; cfg.vqvae.vq_embed_dim=8; cfg.vqvae.num_embed=16
        cfg.vqvae.decoder.hidden_channels=8
        cfg.predictor.moe_hidden=8; cfg.predictor.transformer.d_model=8
        cfg.predictor.transformer.dim_feedforward=16
        cls.base_config=OmegaConf.to_container(cfg,resolve=True)
        cls.checkpoints={}
        for kind in ['legacy','rvq','fsq']:
            config=copy.deepcopy(cls.base_config)
            config['vqvae']['quantizer'].update(research_kind=kind,research_sizes=[4,4],research_levels=[4,4])
            torch.manual_seed(42)
            model=FactorVQVAE(config,T_max=2)
            path=cls.root/f'{kind}.ckpt'
            torch.save({'state_dict':model.state_dict()},path)
            cls.checkpoints[kind]=str(path)
        torch.manual_seed(9)
        cls.batch=torch.randn(7,20,TOTAL_DIM)
        cls.batch[:,:,171:234]=cls.batch[:1,:,171:234]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self,variant):
        cfg=copy.deepcopy(self.__class__.base_config)
        cfg['research']=dict(variant=variant,**copy.deepcopy(NEXT_VARIANTS[variant]))
        kind=cfg['research'].get('quantizer_kind','legacy')
        cfg['vqvae']['quantizer'].update(research_kind=kind,research_sizes=[4,4],research_levels=[4,4])
        cfg['predictor']['saved_model']=self.checkpoints[kind]
        return cfg

    def model(self,variant):
        torch.manual_seed(0)
        return NextPredictor(self.config(variant),T_max=2)

    def test_default_constructor_and_shared_stage1_initialization_are_preserved(self):
        results={}
        for kind in ('legacy','rvq','fsq'):
            cfg=copy.deepcopy(self.__class__.base_config)
            cfg['vqvae']['quantizer'].update(research_kind=kind,research_sizes=[4,4],research_levels=[4,4])
            torch.manual_seed(42)
            model=VQVAE(cfg)
            results[kind]=(model.state_dict(),torch.get_rng_state())
        for kind in ('rvq','fsq'):
            self.assertTrue(torch.equal(results[kind][1],results['legacy'][1]))
            for name,value in results['legacy'][0].items():
                if not name.startswith('quantizer.'):
                    self.assertTrue(torch.equal(value,results[kind][0][name]),name)
        cfg=self.config('baseline')
        torch.manual_seed(0); old=ScreeningPredictor(copy.deepcopy(cfg),T_max=2).eval()
        new=self.model('baseline').eval()
        for name,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[name]),name)

    def test_every_variant_has_finite_gradients_and_restores_exactly(self):
        for variant in NEXT_VARIANTS:
            with self.subTest(variant=variant):
                model=self.model(variant).train()
                parts=unpack_batch(self.batch)
                with torch.no_grad():
                    h=model.encoder(model.revin(parts.stock_feature,mode='norm'))
                    z=model.quantizer(h)[0]
                batch={'canonical':self.batch,'frozen_h':h,'frozen_zq':z,
                       'code_history':z[:,None].expand(-1,20,-1).clone(),
                       'history_mask':torch.ones(7,20),
                       'prev_canonical':self.batch.flip(0), 'prev_frozen_h':h.flip(0), 'prev_frozen_zq':z.flip(0),
                       'align_current':torch.arange(7).float(),'align_previous':torch.arange(6,-1,-1).float()}
                model.log=lambda *a,**kw:None; model.log_dict=lambda *a,**kw:None
                loss=model.training_step(batch,0)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name,param in model.named_parameters():
                    if name.startswith(('encoder.','quantizer.','revin.')):
                        self.assertIsNone(param.grad,name)
                    elif param.grad is not None:
                        self.assertTrue(torch.isfinite(param.grad).all(),name)
                if variant.startswith('history'):
                    self.assertGreater(float(model.history_head.weight.grad.abs().sum()),0)
                restored=self.model(variant)
                restored.load_state_dict(model.state_dict(),strict=True)
                left=model.eval().batch_prediction(batch)[1][0]
                right=restored.eval().batch_prediction(batch)[1][0]
                self.assertTrue(torch.equal(left,right))

    def test_soft_router_prediction_path_has_nontrivial_gradient(self):
        model=self.model('soft_route').eval()
        moe=model.loadings.fusion.moe
        moe.W_h.data.copy_(torch.tensor([[1.3,-.2],[.1,.8]]))
        z=torch.randn(7,8,requires_grad=True)
        gates,_=moe.noisy_top_k_gating(z,False)
        self.assertTrue(torch.allclose(gates.sum(1),torch.ones(7)))
        gates[:,0].sum().backward()
        self.assertGreater(float(z.grad.abs().sum()),1e-4)


class ResearchQuantizerTests(unittest.TestCase):
    def test_legacy_factory_rng_and_state_are_exact(self):
        cfg=dict(num_embed=16,vq_embed_dim=8,quantizer=dict(commit_weight=.25,distance='l2',
                  anchor='probrandom',first_batch=False,contras_loss=True))
        torch.manual_seed(42); old=VectorQuantiser(16,8,.25,distance='l2',anchor='probrandom',first_batch=False,contras_loss=True)
        state=torch.get_rng_state()
        torch.manual_seed(42); new=build_quantizer(cfg)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for name,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[name]))

    def test_residual_lookup_and_straight_through(self):
        model=ResidualVectorQuantiser([2,2],2,.25,distance='l2',anchor='probrandom',first_batch=False,contras_loss=False).eval()
        model.coarse.embedding.weight.data.copy_(torch.tensor([[0.,0.],[2.,0.]]))
        model.fine.embedding.weight.data.copy_(torch.tensor([[0.,0.],[0.,1.]]))
        h=torch.tensor([[2.1,.9]],requires_grad=True)
        z,loss,(_,_,idx)=model(h)
        self.assertTrue(torch.allclose(z,torch.tensor([[2.,1.]])))
        self.assertEqual(int(idx[0]),3)
        (z.sum()+loss).backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_fsq_fixed_grid_indices_and_projection_gradients(self):
        model=FiniteScalarQuantiser([4,4],8)
        h=torch.randn(13,8,requires_grad=True)
        z,loss,(_,enc,idx)=model(h)
        self.assertEqual(tuple(enc.shape),(13,16))
        self.assertTrue(((idx>=0)&(idx<16)).all())
        (z.square().mean()+loss).backward()
        self.assertGreater(float(model.to_grid.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.from_grid.weight.grad.abs().sum()),0)


if __name__=='__main__':
    unittest.main()
