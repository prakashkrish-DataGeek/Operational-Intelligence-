"""Metrics and block-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss) if ss > 0 else float("nan")


def wape(y, p):
    """Weighted absolute percentage error: sum|error| / sum|actual|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - p)) / d) if d > 0 else float("nan")


def bias(y, p):
    """Signed total error as a fraction of total actual (positive = over-prediction)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.sum(np.abs(y))
    return float(np.sum(p - y) / d) if d > 0 else float("nan")


def block_bootstrap_ci(y, p, fn, block: int, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """CI for metric `fn` by resampling contiguous blocks (rows are autocorrelated)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    n = len(y)
    if n < 2 * block:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, n, block)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(starts, size=len(starts), replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in pick])
        vals.append(fn(y[idx], p[idx]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))
