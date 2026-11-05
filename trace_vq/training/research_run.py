"""Epoch records with complete RNG state for serial research fit recovery."""
import random
import time

import numpy as np
import torch

from trace_vq.cli.screening import EpochRecord


def capture_rng(device="cpu"):
    numpy = np.random.get_state()
    numpy = (numpy[0], numpy[1].tolist(), *numpy[2:])
    return dict(python=random.getstate(), numpy=numpy,
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None)


def restore_rng(state):
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


class RecoverableEpochRecord(EpochRecord):
    """No new random draws; restore before Lightning prefetches shuffled days."""
    def __init__(self, output, device="cpu"):
        super().__init__(output)
        self.device = device
        self.previous_fit_seconds = 0.0
        self.previous_peak_allocated = 0.0
        self.previous_peak_reserved = 0.0
        self.fit_start = time.perf_counter()
        self.pending_rng = None

    def state_dict(self):
        cuda = self.device.startswith("cuda")
        return dict(rng=capture_rng(self.device), rows=self.rows,
                    elapsed_seconds=time.monotonic() - self.start,
                    fit_seconds=self.previous_fit_seconds + time.perf_counter() - self.fit_start,
                    peak_allocated=max(self.previous_peak_allocated,
                                       torch.cuda.max_memory_allocated() / 2**20 if cuda else 0),
                    peak_reserved=max(self.previous_peak_reserved,
                                      torch.cuda.max_memory_reserved() / 2**20 if cuda else 0))

    def load_state_dict(self, state_dict):
        self.rows = state_dict["rows"]
        self.previous_fit_seconds = state_dict["fit_seconds"]
        self.previous_peak_allocated = state_dict["peak_allocated"]
        self.previous_peak_reserved = state_dict["peak_reserved"]
        self.start = time.monotonic() - state_dict["elapsed_seconds"]
        self.pending_rng = state_dict["rng"]

    def on_fit_start(self, trainer, pl_module):
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
