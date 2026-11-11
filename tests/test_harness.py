"""Harness tests on made-up data: do the splits, metrics and checks behave?"""
import numpy as np
import pandas as pd
import pytest

from kasauti import data, metrics, production, report, subsurface


def test_metrics_known_values():
    y, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
    assert metrics.mae(y, p) == pytest.approx(7 / 3)
    assert metrics.wape(y, p) == pytest.approx(7 / 60)
    assert metrics.bias(y, p) == pytest.approx(3 / 60)
    assert metrics.r2(y, y) == 1.0


def test_gardner_matches_hand_calculation():
    # rho 2.65 g/cc -> Vp = (2.65/0.31)**4 = 5340 m/s -> 304800/5340 = 57.08 us/ft
    assert subsurface.gardner_dt(np.array([2.65]))[0] == pytest.approx(57.08, abs=0.01)


def test_conformal_quantile_is_finite_sample_corrected():
    r = np.arange(1, 10)  # 9 residuals; ceil(10 * 0.9) = 9th smallest
    assert subsurface.conformal_q(r, 0.9) == 9
    assert subsurface.conformal_q(np.arange(1, 100), 0.9) == 90


def test_lowo_never_trains_on_the_test_well(synthetic_logs):
    df = subsurface.prepare(synthetic_logs, "DT")
    res = subsurface.lowo(df, "DT")
    assert len(res["folds"]) == 3
    for f in res["folds"]:
        assert f["test_well"] not in f["train_wells"]
        assert f["n_train"] + f["n_test"] == len(df)


def test_null_target_gives_no_skill(synthetic_logs):
    """If the target is noise, a blind back-test must not report skill."""
    noise = synthetic_logs.copy()
    noise["DT"] = np.random.default_rng(3).normal(80, 10, len(noise))
    res = subsurface.lowo(subsurface.prepare(noise, "DT"), "DT")
    assert res["pooled"]["gbm"]["r2"] < 0.05


def test_grouped_intervals_cover_on_exchangeable_wells(synthetic_logs):
    same = synthetic_logs.copy()
    same["DT"] = 55 + 130 * same["NPHI"] + np.random.default_rng(4).normal(0, 2, len(same))
    cov = subsurface.lowo(subsurface.prepare(same, "DT"), "DT")["pooled"]["interval"]["grouped_coverage"]
    assert 0.85 <= cov <= 0.95


def test_subsurface_is_deterministic(synthetic_logs):
    assert subsurface.run(synthetic_logs) == subsurface.run(synthetic_logs)


def test_vfm_blocks_never_see_the_future(synthetic_prod):
    g = production.vfm_frame(synthetic_prod)
    g = g[g.WELL == "P-1"]
    blocks = list(production.vfm_blocks(g))
    assert len(blocks) >= 3
    seen = []
    for tr, te in blocks:
        assert tr["DATE"].max() < te["DATE"].min()
        assert len(tr) >= production.MIN_TRAIN_DAYS
        seen.append(te)
    scored = pd.concat(seen)
    assert scored["DATE"].is_unique


def test_vfm_handles_a_dead_sensor(synthetic_prod):
    res = production.vfm_walk_forward(synthetic_prod)  # AVG_ANNULUS_PRESS is all NaN
    assert res["pooled"]["gbm"]["wape"] < 0.2


def test_forecast_uses_only_history(synthetic_prod):
    """Changing the future must not change any forecast."""
    base = production.forecast_backtest(synthetic_prod)["_forecasts"]
    altered = synthetic_prod.copy()
    cut = pd.Timestamp("2011-03-01")
    altered.loc[altered.DATE >= cut, "BORE_OIL_VOL"] *= 5
    new = production.forecast_backtest(altered)["_forecasts"]
    before = base["origin"] <= cut
    cols = ["persistence_30d", "persistence_7d", "arps_exponential"]
    pd.testing.assert_frame_equal(base.loc[before, cols], new.loc[before, cols])


def test_arps_recovers_a_known_decline():
    idx = pd.date_range("2012-01-01", periods=200, freq="D")
    t = np.arange(200)
    hist = pd.DataFrame({"BORE_OIL_VOL": 1000 * np.exp(-0.003 * t), "ON_STREAM_HRS": 24.0}, index=idx)
    truth = float(np.sum(1000 * np.exp(-0.003 * (199 + np.arange(1, 31)))))
    assert production.arps_exponential(hist, 30) == pytest.approx(truth, rel=1e-6)


def test_tampered_data_is_refused(tmp_path):
    for s in data.MANIFEST:
        (tmp_path / s.name).write_bytes(b"not volve")
    with pytest.raises(data.DataError, match="Checksum mismatch"):
        data.verify(tmp_path)
    with pytest.raises(data.DataError, match="Missing"):
        data.verify(tmp_path / "empty")


def test_compare_flags_a_changed_number():
    old = {"a": {"b": 1.0, "c": [1, 2]}, "environment": {"x": 1}}
    assert report.compare({"a": {"b": 1.0, "c": [1, 2]}, "environment": {"x": 2}}, old) == []
    assert len(report.compare({"a": {"b": 1.001, "c": [1, 2]}}, old)) == 1
