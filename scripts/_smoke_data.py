"""Compact, unmodified real-data windows for a short CLI smoke run."""

import pickle

import numpy as np

from trace_vq.data.schema import unpack_batch


class WindowSampler:
    """Keep complete daily cross-sections without serializing the full source."""

    def __init__(self, windows, index):
        unpack_batch(windows)
        if len(windows) != len(index) or not index.is_unique:
            raise ValueError("Sample windows and their unique index must align")
        self.windows = windows
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        return self.windows[index]

    def get_index(self):
        return self.index


def sample_real_windows(path, days):
    """Read the existing pickle and keep its first complete trading days."""
    with path.open("rb") as stream:
        source = pickle.load(stream)
    index = source.get_index()
    dates = index.get_level_values("datetime")
    selected_dates = dates.unique().sort_values()[:days]
    if len(selected_dates) != days:
        raise ValueError(f"Dataset {path} has fewer than {days} trading days")
    positions = np.flatnonzero(dates.isin(selected_dates))
    # The production collator casts to FP32; retain exactly those model inputs.
    windows = np.ascontiguousarray(source[positions], dtype=np.float32)
    for position in sorted({0, len(positions) // 2, len(positions) - 1}):
        np.testing.assert_array_equal(windows[position],
                                      source[int(positions[position])].astype(np.float32))
    sampler = WindowSampler(windows, index[positions])
    info = {
        "source": str(path), "source_rows": len(source), "sampled_rows": len(sampler),
        "days": days, "start": str(selected_dates[0].date()),
        "end": str(selected_dates[-1].date()), "shape": list(windows.shape),
        "sampled_inputs_match_source": True,
    }
    return sampler, info
