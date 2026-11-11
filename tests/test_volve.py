"""Checks on the real data and on the published numbers."""
import json
from pathlib import Path

import pytest

from kasauti import data, production, report, subsurface

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.volve


def test_production_frame_is_clean(volve):
    p = volve["prod"]
    assert set(p["WELL"]) == set(data.PRODUCERS)
    assert not p.duplicated(["WELL", "DATE"]).any()
    assert p["ON_STREAM_HRS"].dropna().between(0, 24).all()
    assert (p["BORE_OIL_VOL"].dropna() >= 0).all()
    assert p["DATE"].min().year == 2008 and p["DATE"].max().year == 2016


def test_field_oil_total_matches_source(volve):
    # Independent check: per-well sums read straight from the workbook, bypassing the loader, add to 9,995,919.93 Sm3.
    assert volve["prod"]["BORE_OIL_VOL"].sum() == pytest.approx(9995919.93, abs=0.5)


def test_logs_have_three_wells_with_sonic(volve):
    df = subsurface.prepare(volve["logs"], "DT")
    assert df.groupby("WELL").size().to_dict() == {"15/9-F-1 A": 10224, "15/9-F-1 B": 3262, "15/9-F-11 A": 11464}
    assert df["DT"].between(40, 200).all()


def test_every_scored_day_is_after_its_training_days(volve):
    g = production.vfm_frame(volve["prod"])
    n = 0
    for _, w in g.groupby("WELL"):
        for tr, te in production.vfm_blocks(w):
            assert tr["DATE"].max() < te["DATE"].min()
            n += len(te)
    assert n == 6991


def test_published_results_reproduce(volve):
    published = json.loads((ROOT / "results" / "results.json").read_text())
    fresh = json.loads(json.dumps(report.run_all(ROOT / "data" / "raw", quiet=True)))
    if fresh["environment"] != published["environment"]:
        pytest.skip("library versions differ from the published run; use `kasauti reproduce` to see drift")
    assert report.compare(fresh, published) == []
