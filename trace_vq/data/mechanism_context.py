"""Past-only representation controls and training-only market-state weights."""
import hashlib

import numpy as np

from trace_vq.data.research_context import CodeHistory
from trace_vq.data.schema import GROUP_SLICES


def date_derangement(length, date):
    """Sattolo permutation: another stock on the same date, with private RNG."""
    permutation = np.arange(length, dtype=np.int64)
    seed = int.from_bytes(hashlib.sha256(f"history-control-v1|{date}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    for end in range(length - 1, 0, -1):
        other = int(rng.integers(end))
        permutation[end], permutation[other] = permutation[other], permutation[end]
    return permutation


class RepresentationHistory(CodeHistory):
    def __init__(self, source, history_root, cache_root, universe, split,
                 feature="quantized", mismatch=False):
        super().__init__(source, history_root, cache_root, universe, split)
        if feature not in ("continuous", "quantized"):
            raise ValueError("Unknown historical representation")
        self.feature = 0 if feature == "continuous" else 1
        self.mismatch = mismatch
        self.dates = source.get_index().get_level_values("datetime").to_numpy()

    def __getitem__(self, positions):
        result = self.source[positions]
        past = self.positions[self.start + np.asarray(positions)]
        values = np.zeros((*past.shape, self.caches[0].shape[-1]), dtype=np.float32)
        for i, cache in enumerate(self.caches):
            mask = (past >= self.offsets[i]) & (past < self.offsets[i + 1])
            if mask.any():
                values[mask] = cache[past[mask] - self.offsets[i], self.feature]
        mask = (past >= 0).astype(np.float32)
        if self.mismatch and np.ndim(positions):
            dates = self.dates[positions]
            if not np.all(dates == dates[0]):
                raise ValueError("History mismatch requires a complete same-date batch")
            permutation = date_derangement(len(positions), str(dates[0]))
            values, mask = values[permutation], mask[permutation]
            if len(positions) == 1:
                values.fill(0); mask.fill(0)
        result.update(code_history=values, history_mask=mask)
        return result


def state_groups(trend, volatility, thresholds):
    trend, volatility = np.asarray(trend), np.asarray(volatility)
    if not np.isfinite(trend).all() or not np.isfinite(volatility).all():
        raise ValueError("Market-state features must be finite")
    return ((trend >= thresholds[0]).astype(np.int64) * 2 +
            (volatility >= thresholds[1]).astype(np.int64))


def fit_state_balance(trend, volatility):
    """Fit on training dates only; bounded inverse frequencies, mean weight1."""
    trend, volatility = np.asarray(trend), np.asarray(volatility)
    if not len(trend) or len(trend) != len(volatility):
        raise ValueError("Training state observations must be nonempty and aligned")
    thresholds = [float(np.quantile(trend, .5)), float(np.quantile(volatility, .8))]
    groups = state_groups(trend, volatility, thresholds)
    counts = np.bincount(groups, minlength=4)
    probabilities = counts / len(groups)
    raw = np.divide(1., 4 * probabilities, out=np.ones(4), where=probabilities > 0)
    # A common scale followed by clipping enforces BOTH bounds and mean1.
    lo, hi = 0., 8.
    for _ in range(80):
        scale = (lo + hi) / 2
        if np.dot(probabilities, np.clip(scale * raw, .5, 2.)) < 1:
            lo = scale
        else:
            hi = scale
    weights = np.clip((lo + hi) / 2 * raw, .5, 2.)
    return {"thresholds": thresholds, "counts": counts.tolist(), "weights": weights.tolist(),
            "training_days": len(groups), "mean_training_weight": float(np.dot(probabilities, weights)),
            "trend_quantile": .5, "volatility_quantile": .8, "bounds": [.5, 2.],
            "group_definition": "2*strong20dayindextrend + high20dayindexvolatility",
            "feature_positions": [9, 10], "fit_splits": ["train"]}


class StateBalancedDays:
    def __init__(self, source, settings):
        self.source, self.settings = source, settings

    def __len__(self):
        return len(self.source)

    def get_index(self):
        return self.source.get_index()

    def __getitem__(self, positions):
        result = self.source[positions]
        canonical = result["canonical"]
        market = canonical[..., -1, GROUP_SLICES["market"]] if np.ndim(positions) else canonical[-1, GROUP_SLICES["market"]]
        if np.ndim(positions):
            if not np.all(market == market[:1]):
                raise ValueError("State weighting requires a shared same-date market input")
            market = market[0]
        group = int(state_groups(market[9], market[10], self.settings["thresholds"]))
        result.update(state_group=np.float32(group), state_weight=np.float32(self.settings["weights"][group]))
        return result
