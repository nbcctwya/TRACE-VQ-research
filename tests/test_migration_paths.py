"""Historical metadata must select artifacts from the requested source repo."""

from pathlib import Path
import tempfile
import unittest

from scripts.import_hvq_experiments import resolve_source_artifact


class MigrationPathTests(unittest.TestCase):
    def test_relative_and_current_absolute_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            checkpoint = source / "artifacts/028/run/checkpoints/stage1.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"stage1")
            for recorded in (checkpoint, checkpoint.relative_to(source)):
                self.assertEqual(resolve_source_artifact(source, recorded), checkpoint)

    def test_moved_source_wins_even_when_the_historical_checkpoint_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "relocated"
            relative = Path("artifacts/028/run/checkpoints/stage1.ckpt")
            historical = root / "original" / relative
            relocated = source / relative
            for path, content in ((historical, b"old model"), (relocated, b"selected model")):
                path.parent.mkdir(parents=True)
                path.write_bytes(content)
            actual = resolve_source_artifact(source, historical)
            self.assertEqual(actual, relocated)
            self.assertEqual(actual.read_bytes(), b"selected model")

    def test_moved_phase3_archive_resolves_checkpoint_and_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "relocated"
            relative = Path("artifacts/034/phase3/batch/seed1")
            for leaf in ("checkpoints/stage2.ckpt", "res/run/1_best.pkl"):
                path = source / relative / leaf
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(leaf.encode())
            directory = resolve_source_artifact(source, root / "original" / relative)
            for leaf in ("checkpoints/stage2.ckpt", "res/run/1_best.pkl"):
                self.assertEqual((directory / leaf).read_bytes(), leaf.encode())

    def test_missing_relocated_artifact_does_not_fall_back_to_another_repo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root / "original/artifacts/028/run/checkpoints/stage1.ckpt"
            historical.parent.mkdir(parents=True)
            historical.write_bytes(b"foreign model")
            with self.assertRaises(FileNotFoundError):
                resolve_source_artifact(root / "relocated", historical)
