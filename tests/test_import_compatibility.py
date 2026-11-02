"""Legacy import/pickle compatibility and canonical package boundaries."""
import ast
import importlib
import pickle
import unittest
from pathlib import Path

from trace_vq.data.market import CanonicalSampler
from trace_vq.models.confidence.adapter import QuantizationConfidenceAdapter
from trace_vq.training.prediction import GenerateReturn


class ImportCompatibilityTests(unittest.TestCase):
    def test_old_imports_resolve_to_the_same_classes(self):
        pairs = [
            ("module.layers.encoder", "trace_vq.models.temporal.encoder", "SpatialEncoder"),
            ("module.layers.moe", "trace_vq.models.experts.shared_routed", "FactorGatedMoE"),
            ("module.quantise", "trace_vq.models.quantization.codebook", "VectorQuantiser"),
            ("trainer.train_ypred", "trace_vq.training.prediction", "GenerateReturn"),
            ("dataset.market", "trace_vq.data.market", "CanonicalSampler"),
            ("utils.test", "trace_vq.evaluation.inference", "run_inference"),
        ]
        for old, new, name in pairs:
            self.assertIs(getattr(importlib.import_module(old), name),
                          getattr(importlib.import_module(new), name))

    def test_legacy_pickle_global_is_resolvable(self):
        self.assertIs(pickle.loads(b"cdataset.market\nCanonicalSampler\n."), CanonicalSampler)
        self.assertIs(pickle.loads(b"ctrainer.train_ypred\nGenerateReturn\n."), GenerateReturn)

    def test_confidence_adapter_retains_flat_checkpoint_keys(self):
        adapter = QuantizationConfidenceAdapter(128)
        self.assertEqual(set(adapter.state_dict()), {"weight", "bias"})
        self.assertEqual(tuple(adapter.weight.shape), (128, 1))

    def test_package_has_no_dependency_on_old_import_roots(self):
        root = Path(__file__).resolve().parents[1] / "trace_vq"
        forbidden = {"module", "trainer", "dataset", "utils", "runtime"}
        for path in root.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    self.assertNotIn(node.module.split(".")[0], forbidden, str(path))
                if isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertNotIn(name.name.split(".")[0], forbidden, str(path))
