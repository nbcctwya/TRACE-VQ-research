"""Causal cache and routing properties for checklist batches03/04."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.fsq_history import history_features
from trace_vq.data.schema import TOTAL_DIM
from trace_vq.models.batch34 import BATCH03,BATCH04,VARIANTS,Batch34Predictor
from trace_vq.models.experts.routing_research import ResearchMoE
from trace_vq.models.experts.shared_routed import FactorGatedMoE
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.representation import FactorVQVAE


class HistoryTests(unittest.TestCase):
    def build(self,root,index,q,h,train,calendar):
        history_features(index,q,lambda rows:h[rows],train,calendar,root)
        return {p.stem:np.load(p) for p in Path(root).glob("*.npy")}

    def test_future_heldout_changes_cannot_change_past_or_transition_fit(self):
        calendar=pd.date_range("2020-01-01",periods=30)
        idx=pd.MultiIndex.from_product([calendar,list("ABC")],names=["datetime","instrument"])
        rng=np.random.default_rng(9);q=rng.integers(0,8,(len(idx),3),dtype=np.uint8);h=rng.normal(size=(len(idx),8)).astype("float32")
        train=np.arange(len(idx))<45
        with tempfile.TemporaryDirectory() as folder:
            a=self.build(Path(folder)/"a",idx,q,h,train,calendar)
            q2=q.copy();h2=h.copy();q2[66:]=(q2[66:]+3)%8;h2[66:]+=100
            b=self.build(Path(folder)/"b",idx,q2,h2,train,calendar)
            for key in a:
                if key=="histogram": np.testing.assert_array_equal(a[key][:22],b[key][:22])
                else: np.testing.assert_array_equal(a[key][:66],b[key][:66])
            sa=json.loads((Path(folder)/"a/feature_spec.json").read_text())
            sb=json.loads((Path(folder)/"b/feature_spec.json").read_text())
            self.assertEqual(sa["transition_counts"],sb["transition_counts"])

    def test_prototypes_exclude_same_day_and_ngrams_exclude_current_code(self):
        calendar=pd.date_range("2020-01-01",periods=4)
        idx=pd.MultiIndex.from_product([calendar,list("AB")],names=["datetime","instrument"])
        q=np.zeros((8,3),dtype=np.uint8);h=np.repeat(np.array([1,3,100,200,7,9,10,12],dtype=np.float32)[:,None],4,1)
        with tempfile.TemporaryDirectory() as folder:
            a=self.build(Path(folder)/"a",idx,q,h,np.arange(8)<4,calendar)
            np.testing.assert_array_equal(a["prototype"][:2],0)
            np.testing.assert_array_equal(a["prototype"][2:4],2)
            q[6:]=7
            b=self.build(Path(folder)/"b",idx,q,h,np.arange(8)<4,calendar)
            np.testing.assert_array_equal(a["ngram2"][6:],b["ngram2"][6:])
            np.testing.assert_array_equal(a["ngram3"][6:],b["ngram3"][6:])

    def test_missing_prior_market_day_is_masked_and_dwell_resets(self):
        calendar=pd.date_range("2020-01-01",periods=5)
        idx=pd.MultiIndex.from_tuples([(d,s) for i,d in enumerate(calendar) for s in "AB" if not(i==3 and s=="B")],
                                     names=["datetime","instrument"])
        q=np.zeros((len(idx),3),dtype=np.uint8);h=np.ones((len(idx),4),dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            a=self.build(folder,idx,q,h,np.ones(len(idx),bool),calendar)
            row=idx.get_loc((calendar[4],"B"))
            self.assertEqual(a["delta"][row,9],0)
            np.testing.assert_array_equal(a["delta"][row,:3],0)
            self.assertEqual(a["ngram2"][row,0],0)
            np.testing.assert_allclose(a["dwell"][row],np.log(2)/np.log(253))


class RoutingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(7)
    def base(self):return FactorGatedMoE(8,64,hidden_size=64,num_experts=2,k=1,use_shared_expert=True)

    def test_expert_choice_fixed_capacity_and_shared_zero_route_fallback(self):
        base=self.base()
        for p in base.gate.parameters():p.data.zero_()
        base.shared_expert.net[-1].bias.data.fill_(1)
        model=ResearchMoE(base,dict(routing="expert_choice",capacity_factor=1)).eval()
        x,z=torch.randn(7,64),torch.randn(7,8)
        gates,_=model.gates(z)
        self.assertEqual((gates>0).sum(0).tolist(),[4,4])
        output,loss=model(x,z)
        empty=(gates>0).sum(-1)==0
        self.assertTrue(empty.any())
        self.assertTrue(torch.equal(output[empty],base.shared_expert(x[empty])))

    def test_bias_changes_selection_only_and_eval_never_updates_bias(self):
        base=self.base()
        for p in base.gate.parameters():p.data.zero_()
        base.gate[-1].bias.data.copy_(torch.tensor([3.,0.]))
        model=ResearchMoE(base,dict(routing="bias_balance",load_ema_decay=.99,bias_step=.001)).eval()
        model.route_bias.copy_(torch.tensor([-1.,1.]))
        old=model.route_bias.clone();z=torch.zeros(5,8)
        gates,_=model.gates(z)
        probability=torch.softmax(torch.tensor([3.,0.]),0)[1]
        self.assertTrue(torch.equal(gates[:,0],torch.zeros(5)))
        torch.testing.assert_close(gates[:,1],(probability/(probability+1e-6)).expand(5))
        self.assertTrue(torch.equal(old,model.route_bias))
        restored=ResearchMoE(self.base(),model.settings).eval();restored.load_state_dict(model.state_dict(),strict=True)
        self.assertTrue(torch.equal(restored.route_bias,model.route_bias))

    def test_finegrained_routed_active_capacity_remains_near_baseline(self):
        baseline=sum(p.numel() for p in self.base().experts[0].parameters())
        for experts,width,k in ((8,16,4),(16,8,8)):
            model=ResearchMoE(self.base(),dict(routing="finegrained",experts=experts,expert_width=width,active_experts=k))
            counts=model.parameter_counts()
            self.assertEqual(len(model.base.experts),experts)
            self.assertLess(abs(k*counts["routed_expert_parameters_each"]/baseline-1),.1)

    def test_soft_slots_are_permutation_equivariant_and_gradients_flow(self):
        model=ResearchMoE(self.base(),dict(routing="soft_slots",slots_per_expert=4)).eval()
        x,z=torch.randn(7,64,requires_grad=True),torch.randn(7,8,requires_grad=True)
        expected=model(x,z)[0];permutation=torch.tensor([2,0,6,1,4,3,5])
        actual=model(x[permutation],z[permutation])[0]
        torch.testing.assert_close(expected[permutation],actual,atol=1e-6,rtol=1e-6)
        expected.square().mean().backward()
        self.assertGreater(float(model.slot_queries.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(x.grad).all())


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        if not OmegaConf.has_resolver("half"):OmegaConf.register_new_resolver("half",lambda x:int(x)//2)
        cfg=OmegaConf.load(Path(__file__).resolve().parents[1]/"configs/config.yaml")
        cfg.vqvae.hidden_size=8;cfg.vqvae.vq_embed_dim=8;cfg.vqvae.decoder.hidden_channels=8
        cfg.predictor.moe_hidden=8;cfg.predictor.transformer.d_model=8;cfg.predictor.transformer.dim_feedforward=16
        cls.config=OmegaConf.to_container(cfg,resolve=True);cls.checkpoints={}
        for family in ("vq","fsq"):
            c=copy.deepcopy(cls.config)
            if family=="fsq":c["vqvae"]["quantizer"].update(research_kind="fsq",research_levels=[8,8,8])
            torch.manual_seed(42);s1=FactorVQVAE(c,T_max=2)
            path=cls.root/(family+".ckpt");torch.save(dict(state_dict=s1.state_dict()),path);cls.checkpoints[family]=str(path)
        torch.manual_seed(11);cls.canonical=torch.randn(7,20,TOTAL_DIM)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def cfg(self,variant):
        c=copy.deepcopy(self.config);s=VARIANTS[variant]
        c["research"]=dict(variant=variant,**s)
        if s["family"]=="fsq":c["vqvae"]["quantizer"].update(research_kind="fsq",research_levels=[8,8,8])
        c["predictor"]["saved_model"]=self.checkpoints[s["family"]]
        if s.get("routing")=="all_experts":c["predictor"]["k"]=2
        if s.get("routing")=="fsq_sort":c["predictor"]["n_expert"]=3
        if s.get("routing")=="finegrained":c["predictor"].update(n_expert=s["experts"],k=s["active_experts"])
        return c

    def batch(self,model,variant):
        from trace_vq.data.schema import unpack_batch
        parts=unpack_batch(self.canonical)
        with torch.no_grad():
            h=model.encoder(model.revin(parts.stock_feature,mode="norm"));z=model.quantizer(h)[0]
        batch=dict(canonical=self.canonical,frozen_h=h,frozen_zq=z)
        if variant in ("code_delta","code_dwell","code_trans","code_vol","proto_memory"):
            from trace_vq.data.fsq_history import FEATURE_DIMS
            batch["context_feature"]=torch.randn(7,8 if variant=="proto_memory" else FEATURE_DIMS[variant])
            if variant=="proto_memory":batch["context_mask"]=torch.ones(7,1)
        if variant.startswith("fsq_gram"):batch["gram_id"]=torch.arange(1,8)[:,None].float()
        if variant=="code_hist_state":batch["state_hist"]=torch.randn(1,27).expand(7,-1).clone()
        return batch

    def test_all_fifteen_variants_backward_frozen_stage1_and_strict_restore(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                torch.manual_seed(0);model=Batch34Predictor(self.cfg(variant),T_max=2).eval()
                batch=self.batch(model,variant)
                model.train();model.log=lambda *a,**kw:None;model.log_dict=lambda *a,**kw:None
                loss=model.training_step(batch,0);loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name,p in model.named_parameters():
                    if name.startswith(("encoder.","quantizer.","revin.")):self.assertIsNone(p.grad)
                    elif p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all(),name)
                model.eval();expected=model.batch_prediction(batch)[1][0]
                restored=Batch34Predictor(self.cfg(variant),T_max=2).eval();restored.load_state_dict(model.state_dict(),strict=True)
                self.assertTrue(torch.equal(expected,restored.batch_prediction(batch)[1][0]))

    def test_batch03_zero_branches_preserve_exact_initial_predictions(self):
        for variant in BATCH03:
            with self.subTest(variant=variant):
                cfg=self.cfg(variant)
                torch.manual_seed(0);baseline=ScreeningPredictor(cfg,T_max=2).eval()
                torch.manual_seed(0);model=Batch34Predictor(cfg,T_max=2).eval()
                batch=self.batch(model,variant)
                self.assertTrue(torch.equal(baseline.batch_prediction(batch)[1][0],model.batch_prediction(batch)[1][0]))


if __name__=="__main__":unittest.main()
