from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kasauti import data

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _have_volve() -> bool:
    try:
        data.verify(RAW)
        return True
    except data.DataError:
        return False


HAVE_VOLVE = _have_volve()


def pytest_collection_modifyitems(config, items):
    skip = pytest.mark.skip(reason="Volve files not in data/raw; run `kasauti fetch`")
    for item in items:
        if "volve" in item.keywords and not HAVE_VOLVE:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def synthetic_logs() -> pd.DataFrame:
    """Three made-up wells. Used only to test the harness, never reported."""
    rng = np.random.default_rng(1)
    frames = []
    for i, well in enumerate(["W-1", "W-2", "W-3"]):
        n = 1500
        phi = np.clip(0.18 + 0.08 * np.sin(np.arange(n) / 60 + i) + rng.normal(0, 0.02, n), 0.02, 0.35)
        frames.append(pd.DataFrame({
            "WELL": well, "DEPTH": 2000 + 0.1 * np.arange(n),
            "GR": 40 + 200 * (0.3 - phi) + rng.normal(0, 5, n),
            "RHOB": 2.65 - 1.6 * phi + rng.normal(0, 0.02, n),
            "NPHI": phi + rng.normal(0, 0.02, n),
            "PEF": 3 + rng.normal(0, 0.3, n),
            "RT": np.exp(rng.normal(1, 0.5, n)),
            "CALI": 8.5,
            "DT": 55 + 130 * phi + rng.normal(0, 2, n) + 3 * i,
            "DTS": 100 + 220 * phi + rng.normal(0, 6, n),
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_prod() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    frames = []
    for well, q0 in (("P-1", 1500.0), ("P-2", 800.0)):
        n = 500
        hrs = np.where(rng.random(n) < 0.06, 0.0, 24.0)
        choke = np.clip(60 + rng.normal(0, 10, n), 5, 100)
        whp = 60 - 0.03 * np.arange(n) + rng.normal(0, 1, n)
        oil = q0 * np.exp(-0.002 * np.arange(n)) * (choke / 60) * hrs / 24 * (1 + rng.normal(0, 0.03, n))
        df = pd.DataFrame({"WELL": well, "DATE": pd.date_range("2010-01-01", periods=n, freq="D"),
                           "ON_STREAM_HRS": hrs, "AVG_DOWNHOLE_PRESSURE": 250 - 0.05 * np.arange(n),
                           "AVG_DOWNHOLE_TEMPERATURE": 105.0, "AVG_DP_TUBING": 180.0, "AVG_ANNULUS_PRESS": np.nan,
                           "AVG_CHOKE_SIZE_P": choke, "AVG_WHP_P": whp, "AVG_WHT_P": 80.0, "DP_CHOKE_SIZE": whp - 30,
                           "BORE_OIL_VOL": oil, "BORE_GAS_VOL": oil * 150, "BORE_WAT_VOL": oil * 0.4})
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="session")
def volve():
    hashes = data.verify(RAW)
    return {"hashes": hashes, "prod": data.load_production(RAW), "logs": data.load_logs(RAW), "tops": data.load_tops(RAW)}
