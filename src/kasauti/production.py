"""Track P: production surveillance on real Volve daily data, scored forward in time.

P1  Virtual flow metering: estimate a well's daily oil volume from its sensors.
P2  Short-term forecasting: total oil over the next 30 and 90 days.

Every model is trained only on days strictly before the days it is scored on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .data import SENSORS

SEED = 0
TARGET = "BORE_OIL_VOL"
MIN_TRAIN_DAYS = 180     # on-stream days before the first forecast
BLOCK_DAYS = 90          # recalibration interval (calendar days)
RECENT_DAYS = 365
CAL_DAYS = 30


def _hgb():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, min_samples_leaf=20, random_state=SEED
    )


def _ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), Ridge(alpha=1.0))


def _fit_predict(model, tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Fit on train, predict test, using only sensors that reported at least once in train.

    A failed gauge leaves a column entirely empty for long periods (F-12 downhole gauge).
    """
    live = [c for c in cols if tr[c].notna().any()]
    return model.fit(tr[live].values, tr[TARGET].values).predict(te[live].values)


def vfm_frame(prod: pd.DataFrame) -> pd.DataFrame:
    df = prod[(prod["ON_STREAM_HRS"] > 0) & prod[TARGET].notna()].copy()
    return df.sort_values(["WELL", "DATE"]).reset_index(drop=True)


def vfm_blocks(g: pd.DataFrame):
    """Yield (train, test) with every train date strictly before every test date."""
    g = g.sort_values("DATE")
    if len(g) <= MIN_TRAIN_DAYS:
        return
    start = g["DATE"].iloc[MIN_TRAIN_DAYS]
    end = g["DATE"].iloc[-1]
    while start <= end:
        stop = start + pd.Timedelta(days=BLOCK_DAYS)
        tr, te = g[g.DATE < start], g[(g.DATE >= start) & (g.DATE < stop)]
        if len(te):
            assert tr["DATE"].max() < te["DATE"].min(), "future data leaked into training"
            yield tr, te
        start = stop


def vfm_walk_forward(prod: pd.DataFrame) -> dict:
    df = vfm_frame(prod)
    X = list(SENSORS)
    rows = []
    for well, g in df.groupby("WELL"):
        for tr, te in vfm_blocks(g):
            cal = tr.tail(CAL_DAYS)
            rate = cal[TARGET].sum() / cal["ON_STREAM_HRS"].sum()
            recent = tr[tr.DATE >= tr.DATE.max() - pd.Timedelta(days=RECENT_DAYS)]
            out = te[["WELL", "DATE", TARGET]].copy()
            out["last_calibration"] = rate * te["ON_STREAM_HRS"].values
            out["ridge"] = _fit_predict(_ridge(), tr, te, X)
            out["gbm"] = _fit_predict(_hgb(), tr, te, X)
            out["gbm_recent"] = _fit_predict(_hgb(), recent, te, X)
            out["n_train"] = len(tr)
            rows.append(out)
    pred = pd.concat(rows, ignore_index=True)
    methods = ["last_calibration", "ridge", "gbm", "gbm_recent"]

    def score(d):
        y = d[TARGET].values
        return {m: {"wape": M.wape(y, d[m].values), "mae": M.mae(y, d[m].values),
                    "bias": M.bias(y, d[m].values), "r2": M.r2(y, d[m].values)} for m in methods}

    pooled = score(pred)
    for m in methods:
        pooled[m]["wape_ci95"] = M.block_bootstrap_ci(pred[TARGET].values, pred[m].values, M.wape, block=30, seed=SEED)

    # The leaky protocol, for comparison only: shuffle all days of all wells.
    live = [c for c in X if df[c].notna().any()]
    y, Xa = df[TARGET].values, df[live].values
    p = np.zeros(len(y))
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=SEED).split(Xa):
        p[te] = _hgb().fit(Xa[tr], y[tr]).predict(Xa[te])
    random_split = {"wape": M.wape(y, p), "mae": M.mae(y, p), "r2": M.r2(y, p)}

    return {
        "protocol": f"per-well expanding walk-forward, retrain every {BLOCK_DAYS} days, first {MIN_TRAIN_DAYS} on-stream days held for training",
        "features": X, "n_scored_days": int(len(pred)),
        "scored_period": [str(pred.DATE.min().date()), str(pred.DATE.max().date())],
        "pooled": pooled,
        "per_well": {w: {"n": int(len(g)), **score(g)} for w, g in pred.groupby("WELL")},
        "random_row_split_gbm": random_split,
        "leak_gap": {"wape_random": random_split["wape"], "wape_forward": pooled["gbm"]["wape"],
                     "wape_inflation": pooled["gbm"]["wape"] / random_split["wape"]},
        "_predictions": pred,
    }


# ------------------------------------------------------------------------- forecasting

def daily_series(g: pd.DataFrame) -> pd.DataFrame:
    s = g.set_index("DATE")[[TARGET, "ON_STREAM_HRS"]]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(idx).fillna(0.0)


def arps_exponential(hist: pd.DataFrame, horizon: int) -> float | None:
    """Exponential decline fitted to the last 180 days of full-day rates."""
    w = hist.tail(180)
    full = w[w["ON_STREAM_HRS"] >= 20]
    full = full[full[TARGET] > 0]
    if len(full) < 30:
        return None
    t = (full.index - w.index[-1]).days.values.astype(float)
    q = (full[TARGET] * 24.0 / full["ON_STREAM_HRS"]).values
    slope, intercept = np.polyfit(t, np.log(q), 1)
    d = float(np.clip(-slope, 0.0, 0.02))
    q0 = float(np.exp(intercept))
    uptime = float(hist["ON_STREAM_HRS"].tail(90).mean() / 24.0)
    days = np.arange(1, horizon + 1)
    return float(np.sum(q0 * np.exp(-d * days)) * uptime)


def forecast_backtest(prod: pd.DataFrame, horizons=(30, 90), step: int = 30, min_history: int = 180) -> dict:
    rows = []
    for well, g in prod.groupby("WELL"):
        s = daily_series(g)
        for h in horizons:
            for o in range(min_history, len(s) - h + 1, step):
                hist, fut = s.iloc[:o], s.iloc[o:o + h]
                assert hist.index.max() < fut.index.min(), "future data leaked into history"
                p30 = float(hist[TARGET].tail(30).mean() * h)
                arps = arps_exponential(hist, h)
                rows.append({"WELL": well, "origin": s.index[o], "horizon": h,
                             "actual": float(fut[TARGET].sum()),
                             "persistence_30d": p30,
                             "persistence_7d": float(hist[TARGET].tail(7).mean() * h),
                             "arps_exponential": p30 if arps is None else arps,
                             "arps_fallback": arps is None})
    fc = pd.DataFrame(rows)
    methods = ["persistence_30d", "persistence_7d", "arps_exponential"]
    out = {"protocol": f"rolling origin every {step} days after {min_history} days of history", "horizons": {}}
    for h, d in fc.groupby("horizon"):
        y = d["actual"].values
        res = {}
        for m in methods:
            ape = np.abs(d[m].values - y)[y > 0] / y[y > 0]
            res[m] = {"wape": M.wape(y, d[m].values), "bias": M.bias(y, d[m].values),
                      "median_ape": float(np.median(ape))}
        base = res["persistence_30d"]["wape"]
        for m in methods:
            res[m]["skill_vs_persistence_30d"] = 1 - res[m]["wape"] / base
        out["horizons"][str(h)] = {"n_forecasts": int(len(d)),
                                   "arps_fallback_share": float(d["arps_fallback"].mean()), "methods": res}
    out["_forecasts"] = fc
    return out


def run(prod: pd.DataFrame) -> dict:
    return {"vfm": vfm_walk_forward(prod), "forecast": forecast_backtest(prod)}
