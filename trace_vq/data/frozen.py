"""Exact FP32 caching of a frozen Stage 1, indexed by original dataset rows.

Daily cross-sections remain intact. This avoids recomputing identical frozen
representations in every epoch and retains both h and z_q for the confidence
adapter. Source datasets are never written.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from trace_vq.data.schema import unpack_batch


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    temporary.write_text(json.dumps(clean(value), indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


class DailyPositions(Sampler):
    """Same date/order/shuffle semantics as DailyBatchSamplerRandom, preindexed."""

    def __init__(self, source, shuffle=False):
        self.shuffle = shuffle
        index = source.get_index()
        groups = pd.Series(np.arange(len(index))).groupby(index.get_level_values("datetime"), sort=True)
        self.positions = list(groups.indices.values())

    def __iter__(self):
        days = np.arange(len(self.positions))
        if self.shuffle:
            np.random.shuffle(days)
        for day in days:
            yield self.positions[day]

    def __len__(self):
        return len(self.positions)


def float_collate(batch):
    # Automatic batching is disabled: Qlib fetches each complete cross-section
    # in one indexed operation. The positional order and FP32 values are the
    # same as stacking individual stock windows, without repeated copies.
    if isinstance(batch, dict):
        return {name: torch.as_tensor(value).float() for name, value in batch.items()}
    return torch.as_tensor(batch).float()


def daily_loader(dataset, shuffle=False, num_workers=0):
    first = dataset[0]
    canonical = first["canonical"] if isinstance(first, dict) else first
    unpack_batch(canonical[None])
    return DataLoader(dataset, sampler=DailyPositions(dataset, shuffle), batch_size=None,
                      num_workers=num_workers, pin_memory=True, collate_fn=float_collate)


class FrozenDataset(Dataset):
    def __init__(self, canonical, cache_path):
        self.canonical = canonical
        self.cache_path = str(cache_path)
        self.values = np.load(cache_path, mmap_mode="r")
        if self.values.dtype != np.float32 or self.values.shape[:2] != (len(canonical), 2):
            raise ValueError("Frozen representation cache does not match the dataset")

    def __len__(self):
        return len(self.canonical)

    def get_index(self):
        return self.canonical.get_index()

    def __getitem__(self, position):
        return {"canonical": self.canonical[position],
                "frozen_h": self.values[position, 0].copy(),
                "frozen_zq": self.values[position, 1].copy()}


def cache_identity(source, checkpoint, config, device):
    source, checkpoint = Path(source), Path(checkpoint)
    root = Path(__file__).resolve().parents[2]
    relevant = [root / "trace_vq/data/frozen.py", root / "trace_vq/data/market.py",
                root / "trace_vq/data/schema.py", root / "trace_vq/models/components/normalization.py",
                root / "trace_vq/models/quantization/codebook.py"]
    relevant.extend(sorted((root / "trace_vq/models/temporal").glob("*.py")))
    relevant.extend(sorted((root / "trace_vq/models/quantization").glob("*.py")))
    return {
        "source": str(source.resolve()), "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "stage1_checkpoint": str(checkpoint.resolve()), "stage1_sha256": sha256(checkpoint),
        "vqvae": config["vqvae"], "matmul_precision": torch.get_float32_matmul_precision(),
        "torch": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
        "implementation": {str(path.relative_to(root)): sha256(path) for path in relevant},
        "method": "whole-day frozen-eval Stage1 FP32 h and z_q, original positional rows, v1",
    }


def prepare_frozen_cache(model, canonical, source, cache_path, config, device="cuda", num_workers=0):
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    identity = cache_identity(source, config["predictor"]["saved_model"], config, device)
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata["identity"] != identity:
            raise RuntimeError(f"Refusing to reuse a stale Stage1 cache: {cache_path}")
        result = FrozenDataset(canonical, cache_path)
        if list(result.values.shape) != metadata["shape"]:
            raise RuntimeError("Cache shape differs from its completion record")
        return result, metadata
    if cache_path.exists() or metadata_path.exists():
        raise RuntimeError(f"Incomplete cache pair; inspect before rebuilding: {cache_path}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".partial.npy")
    shape = (len(canonical), 2, model.vq_embed_dim)
    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
    model.eval().to(device)
    loader = daily_loader(canonical, num_workers=num_workers)
    frozen_state = {name: tensor.detach().clone() for name, tensor in model.quantizer.state_dict().items()}
    rows = 0
    with torch.no_grad():
        for day, (positions, batch) in enumerate(zip(loader.sampler, loader), start=1):
            parts = unpack_batch(batch.to(device))
            h = model.encoder(model.revin(parts.stock_feature, mode="norm"))
            z_q = model.quantizer(h)[0]
            if not torch.isfinite(h).all() or not torch.isfinite(z_q).all():
                raise FloatingPointError("Non-finite frozen Stage1 representation")
            values[positions, 0] = h.cpu().numpy()
            values[positions, 1] = z_q.cpu().numpy()
            rows += len(positions)
            if day % 200 == 0 or day == len(loader):
                print(f"cache {cache_path.stem}: {day}/{len(loader)} days; {rows}/{len(canonical)} rows", flush=True)
    if rows != len(canonical):
        raise RuntimeError("Cached row count differs from the dataset")
    for name, tensor in model.quantizer.state_dict().items():
        if not torch.equal(tensor, frozen_state[name]):
            raise RuntimeError("Stage1 quantizer changed during caching")
    values.flush()
    del values
    temporary.replace(cache_path)
    index = canonical.get_index()
    metadata = {"identity": identity, "shape": list(shape), "dtype": "float32",
                "source_sha256": sha256(source), "cache_sha256": sha256(cache_path),
                "days": len(loader), "rows": len(canonical),
                "start": str(index.get_level_values("datetime").min()),
                "end": str(index.get_level_values("datetime").max()),
                "quantizer_unchanged": True}
    atomic_json(metadata_path, metadata)
    return FrozenDataset(canonical, cache_path), metadata
