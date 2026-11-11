"""Data access: pinned sources, checksum verification and loaders.

Kasauti never ships Volve data. `fetch` downloads the five files it needs from
pinned commits of public mirrors and refuses to continue if a SHA-256 differs
from the manifest. Every result file records these hashes.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_MIRROR = (
    "https://raw.githubusercontent.com/yohanesnuwara/volve-machine-learning/"
    "3b089e0e50fc69207b84616c6cb364b5c5ca2960/"
)


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    sha256: str
    bytes: int

    @property
    def url(self) -> str:
        return _MIRROR + urllib.parse.quote(self.path)


MANIFEST: tuple[Source, ...] = (
    Source("Volve production data.xlsx", "Volve production data.xlsx",
           "39aabef8ccfc71a797ecd0a49848ad7fe79ecd4563560e9318e42519bd324dfe", 2343007),
    Source("Volve_well_picks_modified.csv", "Volve_well_picks_modified.csv",
           "53ed9f062d53a584d05ad9388763a1844ecb5a580e216752dff86fad15e6e1a6", 36016),
    Source("15_9-F-1A.LAS", "well_logs/15_9-F-1A.LAS",
           "fa76ab739c21bee0528d59c1d2f08d3f69f74068c7358dfe5fb653fada4528d1", 11354957),
    Source("15_9-F-1B.LAS", "well_logs/15_9-F-1B.LAS",
           "14f011425a20a5815e1ee73374dc7937cff65021341be4c8ea19587907b53273", 12251857),
    Source("15_9-F-11A.LAS", "well_logs/15_9-F-11A.LAS",
           "e0db4f57a042e4b599f1d9f12df2ab66eb7a609fb055d27711038eb4cc43e9b9", 8687815),
)

PRODUCERS = ("15/9-F-1 C", "15/9-F-11", "15/9-F-12", "15/9-F-14", "15/9-F-15 D")
LOG_WELLS = {"15/9-F-1 A": "15_9-F-1A.LAS", "15/9-F-1 B": "15_9-F-1B.LAS",
             "15/9-F-11 A": "15_9-F-11A.LAS"}

SENSORS = (
    "ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE", "AVG_DP_TUBING",
    "AVG_ANNULUS_PRESS", "AVG_CHOKE_SIZE_P", "AVG_WHP_P", "AVG_WHT_P", "DP_CHOKE_SIZE",
)


class DataError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_dir() -> Path:
    return Path.cwd() / "data" / "raw"


def verify(data_dir: Path | None = None) -> dict[str, str]:
    """Return {file: sha256}; raise DataError if a file is missing or altered."""
    data_dir = Path(data_dir or default_dir())
    out = {}
    for s in MANIFEST:
        p = data_dir / s.name
        if not p.exists():
            raise DataError(f"Missing {p}. Run `kasauti fetch` first.")
        digest = sha256(p)
        if digest != s.sha256:
            raise DataError(
                f"Checksum mismatch for {s.name}: got {digest[:12]}..., expected "
                f"{s.sha256[:12]}... Results would not be comparable with the published ones."
            )
        out[s.name] = digest
    return out


def fetch(data_dir: Path | None = None, quiet: bool = False) -> dict[str, str]:
    data_dir = Path(data_dir or default_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    for s in MANIFEST:
        p = data_dir / s.name
        if p.exists() and sha256(p) == s.sha256:
            continue
        if not quiet:
            print(f"fetching {s.name} ({s.bytes / 1e6:.1f} MB)")
        tmp = p.with_suffix(p.suffix + ".part")
        with urllib.request.urlopen(s.url, timeout=120) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        tmp.replace(p)
    return verify(data_dir)


def manifest_json() -> str:
    return json.dumps([{**s.__dict__, "url": s.url} for s in MANIFEST], indent=2)


# --------------------------------------------------------------------------- loaders

def load_production(data_dir: Path | None = None) -> pd.DataFrame:
    """Daily production for the five oil producers, cleaned but not imputed.

    Cleaning rules (all documented in METHOD.md section 3):
    * keep FLOW_KIND == 'production' rows of the five producer wellbores;
    * a downhole gauge reading of exactly 0 is treated as missing (failed gauge);
    * ON_STREAM_HRS is clipped to [0, 24].
    """
    data_dir = Path(data_dir or default_dir())
    df = pd.read_excel(data_dir / "Volve production data.xlsx", sheet_name="Daily Production Data")
    df = df[(df["FLOW_KIND"] == "production") & df["NPD_WELL_BORE_NAME"].isin(PRODUCERS)].copy()
    df = df.rename(columns={"NPD_WELL_BORE_NAME": "WELL", "DATEPRD": "DATE"})
    df["DATE"] = pd.to_datetime(df["DATE"])
    for c in ("AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE", "AVG_DP_TUBING"):
        df.loc[df[c] == 0, c] = np.nan
    df["ON_STREAM_HRS"] = df["ON_STREAM_HRS"].clip(0, 24)
    keep = ["WELL", "DATE", *SENSORS, "BORE_OIL_VOL", "BORE_GAS_VOL", "BORE_WAT_VOL"]
    return df[keep].sort_values(["WELL", "DATE"]).reset_index(drop=True)


def load_logs(data_dir: Path | None = None) -> pd.DataFrame:
    """Wireline/LWD logs for the three wells that have a measured sonic log."""
    import lasio

    data_dir = Path(data_dir or default_dir())
    frames = []
    for well, fname in LOG_WELLS.items():
        las = lasio.read(str(data_dir / fname))
        df = las.df().reset_index()
        df = df.rename(columns={df.columns[0]: "DEPTH"})
        df.insert(0, "WELL", well)
        frames.append(df)
    cols = ["WELL", "DEPTH", "GR", "RHOB", "NPHI", "PEF", "RT", "CALI", "DT", "DTS"]
    return pd.concat(frames, ignore_index=True)[cols]


def load_tops(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir or default_dir())
    t = pd.read_csv(data_dir / "Volve_well_picks_modified.csv")
    t["WELL"] = t["WELL"].str.replace("NO ", "", regex=False).str.strip()
    t["PICKS"] = t["PICKS"].str.strip()
    return t[["WELL", "PICKS", "DEPTH", "TVD"]].rename(columns={"DEPTH": "MD"})
