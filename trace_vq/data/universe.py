"""Point-in-time stock-pool union without changing the Qlib provider."""

from collections import defaultdict

import pandas as pd
from qlib.data import D


def merge_membership(pools):
    """Merge overlapping inclusive intervals per stock, retaining membership gaps."""
    combined = defaultdict(list)
    for pool in pools:
        for symbol, intervals in pool.items():
            combined[symbol].extend((pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals)
    result = {}
    for symbol in sorted(combined):
        merged = []
        for start, end in sorted(combined[symbol]):
            if start > end:
                raise ValueError(f"Reversed membership interval: {symbol}/{start}/{end}")
            if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
                merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else:
                merged.append((start, end))
        result[symbol] = merged
    return result


def load_union(sources, start, end):
    return merge_membership([
        D.list_instruments(D.instruments(source), start_time=start, end_time=end, as_list=False)
        for source in sources
    ])


def configured_instruments(config):
    """Honor a configured source union in both stock and market handlers."""
    settings = config['data_handler_config']
    sources = config.get('source_universes')
    if sources:
        return load_union(sources, settings['start_time'], settings['end_time'])
    return settings['instruments']


def select_membership(instruments, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return {symbol: [(max(begin, start), min(finish, end)) for begin, finish in spans
                     if begin <= end and finish >= start]
            for symbol, spans in instruments.items()
            if any(begin <= end and finish >= start for begin, finish in spans)}


def daily_membership_counts(instruments, calendar):
    counts = pd.Series(0, index=pd.DatetimeIndex(calendar, name="datetime"), dtype="int64")
    for spans in instruments.values():
        present = pd.Series(False, index=counts.index)
        for start, end in spans:
            present.loc[start:end] = True
        counts += present.astype("int64")
    return counts
