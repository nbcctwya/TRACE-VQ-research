"""Copy the four frozen HVQ experiments and emit relocated evaluation configs.

Run from any cwd with the prism-vq interpreter. This reads the source repository
without checking out branches or modifying data, Git state, or source artifacts.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "025": ("wo_ta", "w/o TA", "gru", True, True),
    "034": ("full", "TRACE-VQ", "temporal-attention", True, True),
    "037": ("wo_ca", "w/o CA", "temporal-attention", False, True),
    "038": ("wo_sr", "w/o SR", "temporal-attention", True, False),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args])


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def merge(target, values):
    for key, value in values.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def resolve_source_artifact(source, recorded_path):
    """Resolve historical artifact paths inside the selected source checkout."""
    source = Path(source).expanduser().resolve()
    recorded = Path(recorded_path).expanduser()
    if not recorded.is_absolute():
        candidates = [source / recorded]
    elif recorded.is_relative_to(source):
        candidates = [recorded]
    else:
        # Receipts and summaries retain the absolute path of the machine that
        # trained the model. Keep their artifact suffix after moving the repo.
        candidates = [source.joinpath(*recorded.parts[position:])
                      for position in range(len(recorded.parts) - 1, -1, -1)
                      if recorded.parts[position] == "artifacts"]
        if not candidates:
            raise ValueError(f"Historical path has no artifact root: {recorded}")
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(source) and candidate.exists():
            return candidate
    raise FileNotFoundError(f"Recorded artifact is missing from {source}: {recorded}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "HVQ-Stock")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if source == ROOT or ROOT.is_relative_to(source):
        raise ValueError("Source and target repositories must be separate")
    entries = yaml.safe_load((source / "experiments/queue.yaml").read_text())["experiments"]
    entries = {e["id"]: e for e in entries if e["id"] in VARIANTS}
    if set(entries) != set(VARIANTS):
        raise ValueError("Missing canonical HVQ experiment definition")
    manifest = {
        "source_root": str(source),
        "source_head": git(source, "rev-parse", "HEAD").decode().strip(),
        "source_status": git(source, "status", "--porcelain").decode(),
        "dataset_root": str(source / "dataset/processed"),
        "experiments": {}, "copied_files": [], "filtered_results": [],
        "source_tracked_hashes": {}, "datasets": {},
    }
    tracked = git(source, "ls-files", "-z").decode().split("\0")
    for relative in tracked:
        if relative and (source / relative).is_file():
            manifest["source_tracked_hashes"][relative] = sha256(source / relative)
    for path in sorted((source / "dataset/processed").rglob("*.pkl")):
        relative = path.relative_to(source).as_posix()
        manifest["datasets"][relative] = {"size": path.stat().st_size, "sha256": sha256(path)}
    print(f"Source pinned; {len(manifest['datasets'])} datasets will be reused", flush=True)

    def copy_file(path, destination=None):
        relative = path.relative_to(source)
        destination = destination or ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            link = os.readlink(path)
            if os.path.isabs(link):
                if Path(link).is_relative_to(source):
                    resolved = Path(link).relative_to(source)
                    link = os.path.relpath(ROOT / resolved, destination.parent)
                elif path.is_file():
                    # W&B debug-core links can point to the user's external
                    # log cache. Materialize available logs into the archive.
                    digest = sha256(path)
                    if destination.exists() and sha256(destination) != digest:
                        raise FileExistsError(destination)
                    if not destination.exists():
                        shutil.copy2(path, destination)
                    if sha256(destination) != digest:
                        raise RuntimeError(f"Copy checksum mismatch: {destination}")
                    manifest["copied_files"].append({"source": relative.as_posix(),
                        "target": destination.relative_to(ROOT).as_posix(),
                        "size": path.stat().st_size, "sha256": digest,
                        "original_symlink": link})
                    return
            if destination.is_symlink() and os.readlink(destination) != link:
                raise FileExistsError(destination)
            if not destination.is_symlink():
                if destination.exists():
                    raise FileExistsError(destination)
                destination.symlink_to(link, target_is_directory=path.is_dir())
            manifest["copied_files"].append({"source": relative.as_posix(),
                "target": destination.relative_to(ROOT).as_posix(), "symlink": link})
            return
        digest = sha256(path)
        if destination.exists() and sha256(destination) != digest:
            raise FileExistsError(f"Refusing to replace changed destination: {destination}")
        if not destination.exists():
            shutil.copy2(path, destination)
        if sha256(destination) != digest:
            raise RuntimeError(f"Copy checksum mismatch: {destination}")
        manifest["copied_files"].append({"source": relative.as_posix(),
            "target": destination.relative_to(ROOT).as_posix(),
            "size": path.stat().st_size, "sha256": digest})

    def copy_tree(path, destination=None):
        destination = destination or ROOT / path.relative_to(source)
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(path.iterdir()):
            target = destination / child.name
            if child.is_symlink() or child.is_file():
                copy_file(child, target)
            elif child.is_dir():
                copy_tree(child, target)

    receipts = {}
    batches = source / "artifacts/_phase3/batches"
    for path in sorted(batches.glob("*/receipts/*.json")):
        receipt = json.loads(path.read_text())
        task = receipt.get("task", {})
        eid = task.get("experiment") or task.get("code", "").removeprefix("code/")
        if eid not in VARIANTS or receipt.get("accepted") is not True:
            continue
        pair = (eid, int(task["seed"]))
        if pair in receipts:
            raise ValueError(f"Conflicting accepted receipts: {pair}")
        receipts[pair] = (path, receipt)

    copied_stage1 = set()
    for eid, (variant, label, encoder, ca, sr) in VARIANTS.items():
        entry = entries[eid]
        commit = entry["commit"]
        if git(source, "rev-parse", entry["branch"]).decode().strip() != commit:
            raise ValueError(f"Branch no longer equals pinned commit: {eid}")
        raw = git(source, "show", f"{commit}:configs/config.yaml")
        raw_path = ROOT / "configs/hvq" / f"{eid}.yaml"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        reference = ROOT / "artifacts/_reference" / eid
        files = git(source, "ls-tree", "-r", "--name-only", commit, "--",
                    "module", "trainer", "utils", "dataset", "configs/config.yaml").decode().splitlines()
        reference_files = {}
        for relative in files:
            blob = git(source, "show", f"{commit}:{relative}")
            path = reference / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
            reference_files[relative] = hashlib.sha256(blob).hexdigest()
        record = next((source / "experiments/records").glob(f"{eid}-*.md"))
        copy_file(record, ROOT / "docs/migration/source_records" / record.name)
        copy_tree(source / "artifacts" / eid)
        summary = json.loads((source / "artifacts" / eid / "run/summary.json").read_text())
        if summary["commit"] != commit:
            raise ValueError(f"Seed0 commit mismatch: {eid}")
        stage1 = resolve_source_artifact(source, summary["stage1_source"]["ckpt"])
        stage1_relative = stage1.relative_to(source).as_posix()
        if stage1_relative not in copied_stage1:
            copy_file(stage1)
            marker = stage1.parent.parent / ".stage1.done"
            if marker.is_file():
                copy_file(marker)
            copied_stage1.add(stage1_relative)
        info = {
            "variant": variant, "label": label, "branch": entry["branch"], "commit": commit,
            "encoder": encoder, "confidence_adapter": ca, "shared_expert": sr,
            "raw_config": raw_path.relative_to(ROOT).as_posix(),
            "reference_root": reference.relative_to(ROOT).as_posix(),
            "reference_files": reference_files,
            "stage1_checkpoint": stage1_relative, "stage1_sha256": sha256(stage1),
            "stage1_source": summary["stage1_source"]["source_id"], "runs": {},
        }
        for seed in range(5):
            if seed == 0:
                directory = source / "artifacts" / eid / "run"
                original = yaml.safe_load(raw)
                merge(original, original.get("stage2_presets", {}).get("csi300", {}))
                original["predictor"]["k"] = original["predictor"]["n_expert"] // 2
                original["artifact_root"] = directory.relative_to(source).as_posix()
                original["predictor"]["saved_model"] = str(stage1)
                checkpoint, = directory.glob("checkpoints/0_*.ckpt")
                prediction, = directory.glob("res/*/0_best.pkl")
                receipt_relative = None
            else:
                receipt_path, receipt = receipts[(eid, seed)]
                task = receipt["task"]
                if task["commit"] != commit or task["stage1_sha256"] != info["stage1_sha256"]:
                    raise ValueError(f"Accepted task provenance mismatch: {eid}/{seed}")
                directory = resolve_source_artifact(source, receipt["archive"])
                checkpoint = directory / receipt["validation"]["checkpoint"]
                prediction = directory / receipt["validation"]["prediction"]
                evidence = json.loads((directory / "training_evidence.json").read_text())
                if evidence["checkpoint_sha256"] != sha256(checkpoint):
                    raise ValueError(f"Accepted checkpoint checksum mismatch: {eid}/{seed}")
                original = evidence["config"]
                if original["train"]["seed"] != seed or evidence["commit"] != commit:
                    raise ValueError(f"Training evidence mismatch: {eid}/{seed}")
                copy_file(receipt_path)
                receipt_relative = receipt_path.relative_to(source).as_posix()
            if original["data"]["universe"] != "csi300":
                raise ValueError("Historical artifacts are CSI300 only")
            config = copy.deepcopy(original)
            config["artifact_root"] = None
            config["vqvae"]["encoder"]["type"] = encoder
            config["predictor"]["shared_expert"] = sr
            config["predictor"]["quantization_confidence_adapter"] = ca
            config["predictor"]["saved_model"] = "${oc.env:TRACE_VQ_ROOT,.}/" + stage1_relative
            shared_data = os.path.relpath(source / "dataset/processed", ROOT)
            config["data"]["data_path"] = "${oc.env:TRACE_VQ_DATA_ROOT," + shared_data + "}"
            config["train"]["seed"] = seed
            config["train"]["project_name"] = "TRACE-VQ"
            config["train"]["wandb_mode"] = "offline"
            # Phase 2 ran locally with high/TF32; accepted Phase 3 runs used
            # RTX 2080 Ti (full FP32). Preserve their effective execution
            # precision when evaluating on the current RTX 5060 Ti.
            config["inference"] = {"matmul_precision": "high" if seed == 0 else "highest"}
            for key, sub in (("save_dir", "checkpoints"), ("save_res", "res")):
                config["train"][key] = "${oc.env:TRACE_VQ_ROOT,.}/" + (directory / sub).relative_to(source).as_posix()
            for preset in config.get("stage2_presets", {}).values():
                preset.get("predictor", {}).pop("saved_model", None)
            config_path = ROOT / checkpoint.relative_to(source).with_suffix(".config.yaml")
            config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            provenance = ROOT / directory.relative_to(source) / f"seed{seed}.source_config.json"
            write_json(provenance, original)
            info["runs"][str(seed)] = {
                "checkpoint": checkpoint.relative_to(source).as_posix(),
                "checkpoint_sha256": sha256(checkpoint),
                "prediction": prediction.relative_to(source).as_posix(),
                "config": config_path.relative_to(ROOT).as_posix(),
                "original_config": provenance.relative_to(ROOT).as_posix(),
                "accepted_receipt": receipt_relative,
                "matmul_precision": config["inference"]["matmul_precision"],
            }
        manifest["experiments"][eid] = info
        print(f"Copied {eid} ({label}), seeds 0-4 and matching Stage 1", flush=True)

    for batch, ids in (("baseline-010-019-025-034-r2", ("025", "034")),
                       ("exp-037-038-r1", ("037", "038"))):
        for eid in ids:
            copy_tree(batches / batch / "evaluation" / eid)
    # Keep an immutable byte-for-byte snapshot, and a four-model filtered view.
    copy_tree(source / "results", ROOT / "artifacts/_hvq_results")
    for folder in ("metrics", "tables"):
        for path in sorted((source / "results" / folder).glob("*.csv")):
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames = reader.fieldnames
                rows = [row for row in reader if row.get("model") in VARIANTS]
            target = ROOT / "results" / folder / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            manifest["filtered_results"].append({"source": path.relative_to(source).as_posix(),
                "target": target.relative_to(ROOT).as_posix(), "rows": len(rows),
                "sha256": sha256(target), "transform": "filter model IDs only; retain numeric text"})
    for eid in VARIANTS:
        copy_file(source / "results/curves/ensemble" / f"csi300_{eid}.csv")
    eval_config = json.loads((source / "results/metadata/eval_config.json").read_text())
    eval_config["models"] = list(VARIANTS)
    eval_config["migration"] = "Copied historical results; no retraining or new backtest"
    write_json(ROOT / "results/metadata/eval_config.json", eval_config)
    write_json(ROOT / "results/metadata/experiment_names.json",
               {eid: info["label"] for eid, info in manifest["experiments"].items()})
    manifest["copied_bytes"] = sum(record.get("size", 0) for record in manifest["copied_files"])
    for relative, digest in manifest["source_tracked_hashes"].items():
        if sha256(source / relative) != digest:
            raise RuntimeError(f"Source tracked file changed: {relative}")
    if manifest["source_head"] != git(source, "rev-parse", "HEAD").decode().strip():
        raise RuntimeError("Source HEAD changed during import")
    if manifest["source_status"] != git(source, "status", "--porcelain").decode():
        raise RuntimeError("Source worktree status changed during import")
    for relative, fingerprint in manifest["datasets"].items():
        if sha256(source / relative) != fingerprint["sha256"]:
            raise RuntimeError(f"Shared dataset changed: {relative}")
    manifest["source_unchanged_after_copy"] = True
    write_json(ROOT / "docs/migration/experiments_manifest.json", manifest)
    write_json(ROOT / "results/metadata/manifest.json", {
        "files": [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "results").rglob("*")) if p.is_file()],
        "source_manifest": "docs/migration/experiments_manifest.json",
    })
    print(f"Imported {len(manifest['copied_files'])} files/links, {manifest['copied_bytes']:,} bytes; source unchanged", flush=True)


if __name__ == "__main__":
    main()
