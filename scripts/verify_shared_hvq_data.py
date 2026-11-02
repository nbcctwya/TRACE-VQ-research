"""Verify all reused CSI300 windows against the accepted training fingerprints."""

import gc
import json
from pathlib import Path
import pickle
import time

from verify_hvq_migration import ROOT, fingerprint, write_json


def main():
    manifest = json.loads((ROOT / "docs/migration/experiments_manifest.json").read_text())
    expected = []
    for info in manifest["experiments"].values():
        for seed in range(1, 5):
            run = info["runs"][str(seed)]
            evidence = ROOT / Path(run["checkpoint"]).parents[1] / "training_evidence.json"
            expected.append(json.loads(evidence.read_text())["data_fingerprint"])
    report = {"passed": False, "datasets_copied": False, "accepted_training_runs": len(expected), "splits": {}}
    started = time.monotonic()
    try:
        for split in ("train", "valid", "test"):
            path = Path(manifest["dataset_root"]) / "CN" / f"csi300_20_h10_{split}.pkl"
            with path.open("rb") as stream:
                dataset = pickle.load(stream)
            observed = fingerprint(dataset, split)
            for record in expected:
                assert observed == record[split], f"Shared {split} data differs from an accepted historical run"
            report["splits"][split] = {"passed": True, "path": str(path),
                                       "rows": observed["n"], "sha256_of_all_windows": observed["sha256"]}
            print(f"{split}: all {observed['n']:,} sampled windows match all 16 accepted runs", flush=True)
            del dataset
            gc.collect()
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(ROOT / "docs/migration/data_consistency_report.json", report)


if __name__ == "__main__":
    main()
