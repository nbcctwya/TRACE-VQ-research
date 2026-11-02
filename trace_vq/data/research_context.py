"""Causal, instrument-aligned context; canonical datasets remain read-only."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .frozen import DailyPositions, atomic_json


class PairedDays(Dataset):
    def __init__(self, source):
        self.source = source
        self.days = DailyPositions(source).positions
        self.day_of_row = np.empty(len(source), dtype=np.int32)
        instruments = source.get_index().get_level_values('instrument').to_numpy()
        self.alignments = []
        for day, rows in enumerate(self.days):
            self.day_of_row[rows] = day
            if day:
                _, current, previous = np.intersect1d(instruments[rows], instruments[self.days[day-1]],
                                                     assume_unique=True, return_indices=True)
            else:
                current = previous = np.array([], dtype=np.int64)
            self.alignments.append((current.astype(np.int32), previous.astype(np.int32)))

    def __len__(self):
        return len(self.source)

    def get_index(self):
        return self.source.get_index()

    def __getitem__(self, positions):
        result = self.source[positions]
        if np.ndim(positions) == 0:
            return result
        day = int(self.day_of_row[positions[0]])
        if not np.all(self.day_of_row[positions] == day):
            raise ValueError('Paired training requires a complete same-day batch')
        current, previous = self.alignments[day]
        prior = self.source[self.days[max(0, day-1)]]
        result.update({'prev_' + key: value for key, value in prior.items()})
        result.update(align_current=current, align_previous=previous)
        return result


def build_history_positions(indices, output, identities, days=20):
    """Index prior calendar days; never use a current/future code or labels."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    completion = output / 'completion.json'
    offsets = np.cumsum([0] + [len(index) for index in indices]).tolist()
    metadata = {'days': days, 'offsets': offsets, 'identities': identities,
                'semantics': 'previous market trading days only, instrument aligned, missing codes masked'}
    if completion.exists():
        if json.loads(completion.read_text()) != metadata:
            raise RuntimeError('Code-history identity changed; choose a new output directory')
        return metadata
    index = indices[0].append(indices[1:])
    if not index.is_unique:
        raise ValueError('Duplicate stock/date index across history splits')
    date_codes, dates = pd.factorize(index.get_level_values('datetime'), sort=True)
    stock_codes, stocks = pd.factorize(index.get_level_values('instrument'), sort=True)
    grid = np.full((len(dates), len(stocks)), -1, dtype=np.int32)
    grid[date_codes, stock_codes] = np.arange(len(index), dtype=np.int32)
    temporary = output / 'positions.partial.npy'
    positions = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.int32, shape=(len(index), days))
    for column, lag in enumerate(range(days, 0, -1)):
        valid = date_codes >= lag
        positions[:, column] = -1
        positions[valid, column] = grid[date_codes[valid] - lag, stock_codes[valid]]
    positions.flush()
    del positions
    temporary.replace(output / 'positions.npy')
    atomic_json(completion, metadata)
    return metadata


class CodeHistory(Dataset):
    def __init__(self, source, history_root, cache_root, universe, split):
        self.source = source
        root = Path(history_root) / universe
        metadata = json.loads((root / 'completion.json').read_text())
        self.offsets = np.array(metadata['offsets'])
        order = ('train', 'valid', 'test')
        self.start = self.offsets[order.index(split)]
        end = self.offsets[order.index(split) + 1]
        if end - self.start != len(source):
            raise ValueError('History index differs from the canonical split')
        self.positions = np.load(root / 'positions.npy', mmap_mode='r')
        self.caches = [np.load(Path(cache_root) / universe / f'{name}.npy', mmap_mode='r') for name in order]
        for i, name in enumerate(order):
            cached = json.loads((Path(cache_root) / universe / f'{name}.json').read_text())
            if cached['source_sha256'] != metadata['identities'][i]['source_sha256']:
                raise ValueError('History cache and feature cache have different source data')
            if cached['identity']['stage1_sha256'] != metadata['identities'][i]['stage1_sha256']:
                raise ValueError('History codes came from a different frozen Stage 1')

    def __len__(self):
        return len(self.source)

    def get_index(self):
        return self.source.get_index()

    def __getitem__(self, positions):
        result = self.source[positions]
        past = self.positions[self.start + np.asarray(positions)]
        values = np.zeros((*past.shape, self.caches[0].shape[-1]), dtype=np.float32)
        for i, cache in enumerate(self.caches):
            mask = (past >= self.offsets[i]) & (past < self.offsets[i+1])
            if mask.any():
                values[mask] = cache[past[mask] - self.offsets[i], 1]
        result.update(code_history=values, history_mask=(past >= 0).astype(np.float32))
        return result
