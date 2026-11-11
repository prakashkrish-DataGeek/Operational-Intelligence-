"""Track S: sonic-log synthesis on real Volve wells, scored blind by well.

Task: predict the measured compressional (DT) and shear (DTS) slowness from the
other logs. The truth is a measurement, not an interpretation. The headline
protocol is leave-one-well-out; the random-row protocol is run only to measure
how much it flatters the same model (the "leak gap").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M

FEATURES = ["GR", "RHOB", "NPHI", "PEF", "LOGRT"]
TARGETS = ["DT", "DTS"]
NOMINAL = 0.90
SEED = 0


def prepare(logs: pd.DataFrame, target: str) -> pd.DataFrame:
    df = logs.copy()
    df["LOGRT"] = np.log10(df["RT"].where(df["RT"] > 0))
    df = df.dropna(subset=[*FEATURES, target])
    return df[["WELL", "DEPTH", *FEATURES, target]].sort_values(["WELL", "DEPTH"]).reset_index(drop=True)


class _ClippedRidge:
    """Ridge on features winsorised at the training 1st/99th percentile."""

    def fit(self, X, y):
        self.lo, self.hi = np.percentile(X, [1, 99], axis=0)
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(np.clip(X, self.lo, self.hi), y)
        return self

    def predict(self, X):
        return self.m.predict(np.clip(X, self.lo, self.hi))


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, min_samples_leaf=20, random_state=SEED
    )


def gardner_dt(rhob: np.ndarray) -> np.ndarray:
    """Gardner et al. (1974): rho = 0.31 * Vp**0.25 (g/cc, m/s) inverted to DT in us/ft."""
    vp = (np.clip(rhob, 1.5, 3.1) / 0.31) ** 4
    return 304800.0 / vp


MODELS = {"ridge": _ClippedRidge, "gbm": _hgb}


def conformal_q(residuals: np.ndarray, nominal: float = NOMINAL) -> float:
    r = np.sort(np.abs(residuals))
    k = int(np.ceil((len(r) + 1) * nominal)) - 1
    return float(r[min(k, len(r) - 1)])


def lowo(df: pd.DataFrame, target: str) -> dict:
    """Leave-one-well-out back-test with well-grouped conformal intervals."""
    wells = sorted(df["WELL"].unique())
    folds, pooled = [], {k: {"y": [], "p": []} for k in ["train_mean", "gardner", *MODELS]}
    for test in wells:
        tr, te = df[df.WELL != test], df[df.WELL == test]
        assert test not in set(tr.WELL), "held-out well leaked into training"
        Xtr, ytr, Xte, yte = tr[FEATURES].values, tr[target].values, te[FEATURES].values, te[target].values
        preds = {"train_mean": np.full(len(te), ytr.mean())}
        if target == "DT":
            preds["gardner"] = gardner_dt(te["RHOB"].values)
        for name, make in MODELS.items():
            preds[name] = make().fit(Xtr, ytr).predict(Xte)

        # Conformal half-widths for the GBM, two ways.
        # (a) naive: random 80/20 row split inside the training wells.
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(tr))
        cut = int(0.8 * len(tr))
        m = _hgb().fit(Xtr[idx[:cut]], ytr[idx[:cut]])
        q_naive = conformal_q(ytr[idx[cut:]] - m.predict(Xtr[idx[cut:]]))
        # (b) grouped: residuals come only from wells the model has not seen.
        res = []
        tw = sorted(tr.WELL.unique())
        for cal in tw:
            a, b = tr[tr.WELL != cal], tr[tr.WELL == cal]
            res.append(b[target].values - _hgb().fit(a[FEATURES].values, a[target].values).predict(b[FEATURES].values))
        q_group = conformal_q(np.concatenate(res))

        err = np.abs(yte - preds["gbm"])
        fold = {
            "test_well": test, "train_wells": tw, "n_train": int(len(tr)), "n_test": int(len(te)),
            "models": {k: {"mae": M.mae(yte, v), "rmse": M.rmse(yte, v), "r2": M.r2(yte, v)} for k, v in preds.items()},
            "interval": {
                "nominal": NOMINAL,
                "naive_halfwidth": q_naive, "naive_coverage": float(np.mean(err <= q_naive)),
                "grouped_halfwidth": q_group, "grouped_coverage": float(np.mean(err <= q_group)),
            },
        }
        fold["models"]["gbm"]["mae_ci95"] = M.block_bootstrap_ci(yte, preds["gbm"], M.mae, block=200, seed=SEED)
        folds.append(fold)
        for k, v in preds.items():
            pooled[k]["y"].append(yte)
            pooled[k]["p"].append(v)
    summary = {}
    for k, d in pooled.items():
        if not d["y"]:
            continue
        y, p = np.concatenate(d["y"]), np.concatenate(d["p"])
        summary[k] = {"mae": M.mae(y, p), "rmse": M.rmse(y, p), "r2": M.r2(y, p),
                      "worst_well_r2": min(f["models"][k]["r2"] for f in folds)}
    n = sum(f["n_test"] for f in folds)
    summary["interval"] = {
        "nominal": NOMINAL,
        "naive_coverage": sum(f["interval"]["naive_coverage"] * f["n_test"] for f in folds) / n,
        "grouped_coverage": sum(f["interval"]["grouped_coverage"] * f["n_test"] for f in folds) / n,
        "worst_well_naive_coverage": min(f["interval"]["naive_coverage"] for f in folds),
        "worst_well_grouped_coverage": min(f["interval"]["grouped_coverage"] for f in folds),
    }
    return {"folds": folds, "pooled": summary}


def random_split(df: pd.DataFrame, target: str) -> dict:
    """The protocol most demos use: shuffle rows from all wells, then cross-validate."""
    X, y = df[FEATURES].values, df[target].values
    p = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(X):
        p[te] = _hgb().fit(X[tr], y[tr]).predict(X[te])
    return {"mae": M.mae(y, p), "rmse": M.rmse(y, p), "r2": M.r2(y, p)}


def run(logs: pd.DataFrame) -> dict:
    out = {"protocol": "leave-one-well-out", "features": FEATURES, "targets": {}}
    for t in TARGETS:
        df = prepare(logs, t)
        res = lowo(df, t)
        res["rows_per_well"] = {w: int(n) for w, n in df.groupby("WELL").size().items()}
        res["random_row_split_gbm"] = random_split(df, t)
        g = res["pooled"]["gbm"]
        res["leak_gap"] = {"mae_random": res["random_row_split_gbm"]["mae"], "mae_blind": g["mae"],
                           "mae_inflation": g["mae"] / res["random_row_split_gbm"]["mae"]}
        out["targets"][t] = res
    return out
