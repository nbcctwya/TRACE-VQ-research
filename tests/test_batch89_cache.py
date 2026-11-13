"""Small real cache creation, component IDs, and interrupted-pair recovery."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tests import test_batch89
from tests.test_batch89_recovery import TinyDays
from trace_vq.cli.batch89 import cached
from trace_vq.models.batch89 import Batch89Stage1,Batch89Predictor


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):test_batch89.PredictorTests.setUpClass()
    @classmethod
    def tearDownClass(cls):test_batch89.PredictorTests.tearDownClass()

    def model(self,folder,variant):
        fixture=test_batch89.PredictorTests('test_all_nineteen_stage1_and_stage2_backward_freeze_and_strict_restore')
        cfg=fixture.cfg(variant);cfg['artifact_root']=str(folder)
        s1=Batch89Stage1(copy.deepcopy(cfg),T_max=2).eval();path=folder/'stage1.ckpt'
        torch.save(dict(state_dict=s1.state_dict()),path);cfg['predictor']['saved_model']=str(path)
        return Batch89Predictor(cfg,T_max=2).eval(),cfg

    def test_group_and_residual_component_codes_are_saved_and_reused(self):
        for variant in ('fsq_group','fsq_res'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);model,config=self.model(root,variant);source=root/'source.pkl';source.write_bytes(b'fixture')
                args=SimpleNamespace(output_dir=root,variant=variant,device='cpu',num_workers=0)
                with patch('trace_vq.cli.batch89.read_dataset',return_value=(TinyDays(),source)):
                    data,metadata=cached(model,config,'train',args)
                    codes=np.load(root/'cache/train_components.npy')
                    self.assertEqual(codes.shape,(12,2));self.assertEqual(codes.dtype,np.int16)
                    self.assertTrue(((codes>=0)&(codes<512)).all());self.assertIn('component_codes',metadata)
                    _,again=cached(model,config,'train',args);self.assertEqual(metadata,again)

    def test_interrupted_array_metadata_pair_is_archived_then_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model,config=self.model(root,'fsq_group');source=root/'source.pkl';source.write_bytes(b'fixture')
            args=SimpleNamespace(output_dir=root,variant='fsq_group',device='cpu',num_workers=0)
            with patch('trace_vq.cli.batch89.read_dataset',return_value=(TinyDays(),source)):
                first,_=cached(model,config,'train',args);values=first.values.copy();del first
                (root/'cache/train.json').unlink();(root/'cache/train_components.json').unlink()
                second,_=cached(model,config,'train',args)
                np.testing.assert_array_equal(values,second.values)
                self.assertTrue(list((root/'cache').glob('incomplete_train_*')))
                self.assertTrue(list((root/'cache').glob('incomplete_components_*')))


if __name__=='__main__':unittest.main()
