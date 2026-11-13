"""Interrupt a real stochastic Stage1 fit and compare full optimizer/RNG resume."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from trace_vq.cli.screening import initialize
from trace_vq.data.frozen import daily_loader
from trace_vq.data.schema import TOTAL_DIM
from trace_vq.models.batch89 import Batch89Stage1,VARIANTS
from trace_vq.training.resilient import AtomicEpochCheckpoint,BatchHeartbeat
from trace_vq.training.research_run import capture_rng


class TinyDays:
    def __init__(self):
        self.values=np.random.default_rng(13).normal(size=(12,20,TOTAL_DIM)).astype('float32')
        self.index=pd.MultiIndex.from_product([pd.date_range('2020-01-01',periods=4),list('ABC')],names=['datetime','instrument'])
    def __len__(self):return len(self.index)
    def get_index(self):return self.index
    def __getitem__(self,positions):return self.values[positions]


class RecoveryTests(unittest.TestCase):
    def test_cpu_stage1_dither_resume_preserves_parameters_and_all_rng(self):self.compare('cpu')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_stage1_dither_resume_preserves_parameters_and_all_rng(self):self.compare('cuda')

    def compare(self,device):
        torch.set_num_threads(1)
        if not OmegaConf.has_resolver('half'):OmegaConf.register_new_resolver('half',lambda x:int(x)//2)
        c=OmegaConf.load(Path(__file__).resolve().parents[1]/'configs/config.yaml')
        c.vqvae.hidden_size=8;c.vqvae.vq_embed_dim=8;c.vqvae.decoder.hidden_channels=8
        c.vqvae.predictor.pred_weight=.0001;c.train.wandb_mode='disabled'
        config=OmegaConf.to_container(c,resolve=True);config['research']=dict(variant='fsq_dither',**VARIANTS['fsq_dither'])
        def fit(folder,epochs,checkpoint=None):
            folder.mkdir(parents=True,exist_ok=True);initialize(42);cfg=copy.deepcopy(config);cfg['artifact_root']=str(folder)
            model=Batch89Stage1(cfg,T_max=16)
            recorder=BatchHeartbeat(folder,device);save=AtomicEpochCheckpoint(dirpath=folder/'checkpoints',monitor='val_loss',save_top_k=1,
                save_last=True,mode='min',save_on_train_epoch_end=True,enable_version_counter=False)
            trainer=pl.Trainer(accelerator='gpu' if device=='cuda' else 'cpu',devices=1,max_epochs=epochs,deterministic=True,logger=False,
                enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,callbacks=[recorder,save])
            trainer.fit(model,train_dataloaders=daily_loader(TinyDays(),shuffle=True),val_dataloaders=daily_loader(TinyDays()),
                ckpt_path=str(checkpoint) if checkpoint else None)
            return {k:v.clone() for k,v in model.state_dict().items()},capture_rng(device)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);expected,rng=fit(root/'full',4)
            fit(root/'split',2);checkpoint=root/'split/checkpoints/last.ckpt'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertEqual(saved['epoch'],1);self.assertTrue(saved['optimizer_states']);self.assertTrue(saved['lr_schedulers'])
            actual,restored_rng=fit(root/'split',4,checkpoint)
            for name,value in expected.items():self.assertTrue(torch.equal(value,actual[name]),name)
            self.assertTrue(torch.equal(rng['torch'],restored_rng['torch']))
            self.assertEqual(rng['numpy'],restored_rng['numpy']);self.assertEqual(rng['python'],restored_rng['python'])
            if device=='cuda':
                for a,b in zip(rng['cuda'],restored_rng['cuda']):self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main()
