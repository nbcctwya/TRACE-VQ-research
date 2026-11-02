"""Bounded-memory CSI800 generation with global train-only Qlib preprocessing.

Raw Alpha158/label shards retain Qlib values and point-in-time membership.
Exact medians/MADs are fitted across all training rows, not independently per
year or source stock pool. Final FP32 storage matches the production collator.
"""

import gc
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.constant import EPS
from qlib.contrib.data.handler import Alpha158
from qlib.data import D
from qlib.data.cache import H
from qlib.data.dataset.processor import CSRankNorm, DropnaLabel

from trace_vq.data.market import CanonicalSampler, market_feature_config
from trace_vq.data.schema import GROUP_SLICES, MARKET_INDICES, STOCK_DIM, dataset_basename, unpack_batch
from trace_vq.data.universe import daily_membership_counts, load_union, select_membership


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def robust_parameters(columns):
    """Fit Qlib's median/(MAD+EPS)*1.4826 exactly, one column at a time."""
    means, scales = [], []
    for values in columns:
        values = np.asarray(values)
        mean = np.nanmedian(values)
        scale = np.asarray([np.nanmedian(np.abs(values - mean))], dtype=values.dtype)
        scale += EPS
        scale *= 1.4826
        means.append(mean)
        scales.append(scale[0])
    mean, scale = np.asarray(means), np.asarray(scales)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError('Non-finite train-only normalization parameters')
    return mean, scale


def normalize(values, mean, scale):
    result = np.clip((values - mean) / scale, -3, 3)
    return np.nan_to_num(result, copy=False, nan=0.0).astype(np.float32)


def process_shard(raw, factors, market, parameters, learning):
    """Apply common inference processors, then the inherited learning processors."""
    dates = raw.index.get_level_values('datetime')
    feature = normalize(raw.iloc[:, :STOCK_DIM].to_numpy(), *parameters['feature'])
    prior = normalize(factors.loc[dates].to_numpy(), *parameters['prior'])
    market_values = market.loc[dates].to_numpy(dtype=np.float32)
    label = raw.iloc[:, STOCK_DIM:].to_numpy()
    columns = pd.MultiIndex.from_tuples(
        [(group, name) for group, names in (
            ('feature', raw.columns[:STOCK_DIM]), ('prior', factors.columns),
            ('market', market.columns.get_level_values(1)), ('label', raw.columns[STOCK_DIM:]))
         for name in names])
    frame = pd.DataFrame(np.concatenate((feature, prior, market_values, label), axis=1),
                         index=raw.index, columns=columns).sort_index()
    if learning:
        frame = DropnaLabel()(frame)
        frame = CSRankNorm(fields_group='label')(frame)
    return frame.astype(np.float32)


def _save_pickle(value, path):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def prepare_union_dataset(output_dir, jkp_path, config, kernels=4):
    from trace_vq.data.prepare import build_factor_matrix, label_config

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = dataset_basename('csi800')
    targets = {split: output_dir / 'CN' / f'{base}_{split}.pkl' for split in ('train', 'valid', 'test')}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError('CSI800 split already exists; use a new output root to rebuild')
    stock = config['data_handler_config']
    provider = Path(config['qlib_init']['provider_uri']).expanduser().resolve()
    sources = config.get('source_universes', ['csi300', 'csi500'])
    fingerprints = {name: file_hash(provider / 'instruments' / f'{name}.txt') for name in sources}
    metadata = []
    for symbol in sorted(load_symbols_from_files(provider, sources)):
        for field in ('open', 'high', 'low', 'close', 'volume', 'vwap'):
            path = provider / 'features' / symbol.lower() / f'{field}.day.bin'
            if path.exists():
                stat = path.stat()
                metadata.append((str(path.relative_to(provider)), stat.st_size, stat.st_mtime_ns))
    identity = {'config': config, 'membership_sha256': fingerprints,
                'calendar_sha256': file_hash(provider / 'calendars/day.txt'),
                'raw_field_metadata_sha256': hashlib.sha256(json.dumps(metadata).encode()).hexdigest(),
                'jkp_sha256': file_hash(jkp_path), 'provider': str(provider), 'qlib': qlib.__version__,
                'method': 'point-in-time union, global train-only median/MAD, FP32 canonical storage v1'}
    signature = hashlib.sha256(json.dumps(identity, default=str, sort_keys=True).encode()).hexdigest()
    cache = output_dir / '_build' / f'csi800_{signature[:16]}'
    cache.mkdir(parents=True, exist_ok=True)
    (cache / 'identity.json').write_text(json.dumps(identity, default=str, indent=2))
    qlib.init(provider_uri=str(provider), region='cn', kernels=kernels,
              expression_cache=None, dataset_cache=None,
              exp_manager={'class': 'MLflowExpManager', 'module_path': 'qlib.workflow.expm',
                           'kwargs': {'uri': str(cache / 'qlib_runs'), 'default_exp_name': 'csi800_generation'}})
    instruments = load_union(sources, stock['start_time'], stock['end_time'])
    calendar = pd.DatetimeIndex(D.calendar(start_time=stock['start_time'], end_time=stock['end_time']))
    expected = daily_membership_counts(instruments, calendar)
    expected.rename('union_count').to_csv(cache / 'daily_membership_counts.csv')
    with (cache / 'union_instruments.txt').open('w') as stream:
        for symbol, spans in instruments.items():
            for start, end in spans:
                stream.write(f'{symbol}\t{start.date()}\t{end.date()}\n')
    fields, names = Alpha158.get_feature_config(None)
    label_fields, label_names = label_config()
    shards = []
    fit_start, fit_end = pd.Timestamp(stock['fit_start_time']), pd.Timestamp(stock['fit_end_time'])
    training_counts = pd.Series(0, index=calendar, dtype='int64')
    n_train = 0
    for year in range(calendar[0].year, calendar[-1].year + 1):
        start, end = max(calendar[0], pd.Timestamp(year, 1, 1)), min(calendar[-1], pd.Timestamp(year, 12, 31))
        path = cache / f'raw_{year}.pkl'
        if path.exists():
            raw = pd.read_pickle(path)
        else:
            print(f'[CSI800] Qlib Alpha158 + labels: {year}', flush=True)
            raw = D.features(select_membership(instruments, start, end), fields + label_fields,
                             start_time=start, end_time=end, freq='day')
            raw.columns = names + label_names
            raw = raw.swaplevel().sort_index()
            _save_pickle(raw, path)
        actual = raw.groupby(level='datetime').size()
        declared = expected.loc[start:end]
        available = actual.reindex(declared.index, fill_value=0)
        if not raw.index.is_unique or (available > declared).any():
            raise ValueError('Qlib returned duplicate or excess stock/date rows')
        # Qlib may omit dates with no underlying data. Retain its native rows,
        # as the CSI300 handler does, and keep the coverage difference visible.
        coverage = pd.DataFrame({'declared': declared, 'available': available, 'missing': declared - available})
        coverage.to_csv(cache / f'availability_{year}.csv')
        fitting = (actual.index >= fit_start) & (actual.index <= fit_end)
        training_counts.loc[actual.index[fitting]] = actual.loc[fitting]
        n_train += int(actual.loc[fitting].sum())
        shards.append(path)
        print(f'[CSI800] cached {year}: {len(raw):,} rows; unavailable stock/dates={int(coverage.missing.sum())}', flush=True)
        del raw
        H.clear()
        gc.collect()

    print(f'[CSI800] fitting shared train-only preprocessing on {n_train:,} rows', flush=True)
    matrix_path = cache / 'train_features.npy'
    training = np.lib.format.open_memmap(matrix_path, mode='w+', dtype=np.float32, shape=(STOCK_DIM, n_train))
    offset = 0
    for path in shards:
        raw = pd.read_pickle(path)
        raw = raw.loc[pd.IndexSlice[fit_start:fit_end, :], :]
        if len(raw):
            training[:, offset:offset + len(raw)] = raw.iloc[:, :STOCK_DIM].to_numpy().T
            offset += len(raw)
        del raw
    if offset != n_train:
        raise RuntimeError('Normalization training row count differs from membership')
    training.flush()
    feature_parameters = robust_parameters(training)
    del training
    gc.collect()
    factors = build_factor_matrix(pd.read_csv(jkp_path, parse_dates=['date']),
                                  "location=='chn' and weighting=='vw_cap' and freq=='daily'", calendar, 20)
    repeat = training_counts.to_numpy()
    prior_parameters = robust_parameters(np.repeat(factors[column].to_numpy(), repeat) for column in factors.columns)
    market_fields, _ = market_feature_config(MARKET_INDICES['csi800'])
    daily_market = D.features(['SH600000'], market_fields, start_time=calendar[0], end_time=calendar[-1])
    daily_market = daily_market.droplevel('instrument').reindex(calendar)
    if daily_market.isna().all().any():
        raise ValueError('Missing required market index data')
    # Mask expressions have the indices' calendar, even when a member stock has
    # no own OHLCV row. Match MarketDataHandler's membership-weighted fitting.
    market_repeat = expected.where((expected.index >= fit_start) & (expected.index <= fit_end), 0).to_numpy()
    market_parameters = robust_parameters(np.repeat(daily_market[column].to_numpy(), market_repeat) for column in daily_market)
    market = pd.DataFrame(normalize(daily_market.to_numpy(), *market_parameters), index=calendar,
                          columns=pd.MultiIndex.from_product([['market'], market_fields]))
    parameters = {'feature': feature_parameters, 'prior': prior_parameters, 'market': market_parameters}
    np.savez(cache / 'normalization.npz', **{f'{group}_{part}': value for group, pair in parameters.items()
                                          for part, value in zip(('median', 'scale'), pair)})
    _save_pickle(market, cache / 'market.pkl')
    processed = {}
    for path in shards:
        raw = pd.read_pickle(path)
        for learning in (True, False):
            target = cache / f'{path.stem}_{"learn" if learning else "infer"}.pkl'
            _save_pickle(process_shard(raw, factors, market, parameters, learning), target)
            processed[(path.stem, learning)] = target
        del raw
        gc.collect()
    report = {'signature': signature, 'identity': identity, 'train_normalization_rows': n_train,
              'cache': str(cache), 'storage_dtype': 'float32', 'splits': {}}
    for split, path in targets.items():
        start, end = map(pd.Timestamp, config['dataset']['segments'][split])
        extended = calendar[max(0, calendar.searchsorted(start) - 20)]
        frames = []
        for raw_path in shards:
            year = int(raw_path.stem.split('_')[-1])
            if extended.year <= year <= end.year:
                frame = pd.read_pickle(processed[(raw_path.stem, split != 'test')])
                frame = frame.loc[pd.IndexSlice[extended:end, :], :]
                frames.append(frame)
        frame = pd.concat(frames).sort_index()
        del frames
        gc.collect()
        print(f'[CSI800] serializing {split}: {len(frame):,} rows incl. lookback', flush=True)
        sampler = CanonicalSampler(frame, market, 'csi800', start, end)
        del frame
        unpack_batch(sampler[[0]])
        if sampler.data_arr.dtype != np.float32:
            raise RuntimeError('Expected FP32 sampler storage')
        # All non-finite history values must be accounted for before training.
        for group in ('feature', 'prior', 'market'):
            if not np.isfinite(sampler.data_arr[:-1, GROUP_SLICES[group]]).all():
                raise ValueError(f'{split}/{group}: non-finite inputs')
        path.parent.mkdir(parents=True, exist_ok=True)
        _save_pickle(sampler, path)
        index = sampler.get_index()
        counts = index.to_frame(index=False).groupby('datetime').size()
        report['splits'][split] = {'path': str(path), 'rows': len(sampler), 'days': len(counts),
                                   'min_daily_stocks': int(counts.min()), 'max_daily_stocks': int(counts.max()),
                                   'sample_shape': list(sampler[0].shape), 'bytes': path.stat().st_size,
                                   'sha256': file_hash(path)}
        (output_dir / 'csi800_build_manifest.json').write_text(json.dumps(report, default=str, indent=2))
        print(f'[CSI800] completed {split}: {report["splits"][split]}', flush=True)
        del sampler, index, counts
        gc.collect()
    return targets


def load_symbols_from_files(provider, sources):
    return {line.split()[0] for name in sources
            for line in (provider / 'instruments' / f'{name}.txt').read_text().splitlines() if line.strip()}
