"""Read-only tools over the Volve data, with an audit trail.

The agent never touches a DataFrame. It calls named tools with JSON arguments;
every call is logged with the source file and its SHA-256, so an answer can be
traced back to the bytes it came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

QUANTITIES = {"oil": "BORE_OIL_VOL", "gas": "BORE_GAS_VOL", "water": "BORE_WAT_VOL"}
SENSOR_NAMES = {
    "wellhead_pressure": ("AVG_WHP_P", "bar"),
    "wellhead_temperature": ("AVG_WHT_P", "degC"),
    "downhole_pressure": ("AVG_DOWNHOLE_PRESSURE", "bar"),
    "downhole_temperature": ("AVG_DOWNHOLE_TEMPERATURE", "degC"),
    "choke_opening": ("AVG_CHOKE_SIZE_P", "%"),
    "annulus_pressure": ("AVG_ANNULUS_PRESS", "bar"),
}
LOG_CURVES = ("GR", "RHOB", "NPHI", "PEF", "RT", "DT", "DTS")


class ToolError(Exception):
    """Raised for a bad argument (unknown well, unknown quantity, bad dates)."""


@dataclass
class ToolResult:
    value: Any
    unit: str
    rows: int
    source: str

    @property
    def has_data(self) -> bool:
        return self.rows > 0 and self.value is not None


@dataclass
class Toolbox:
    production: pd.DataFrame
    logs: pd.DataFrame
    tops: pd.DataFrame
    hashes: dict[str, str] = field(default_factory=dict)
    audit: list[dict] = field(default_factory=list)

    PROD_FILE = "Volve production data.xlsx"
    TOPS_FILE = "Volve_well_picks_modified.csv"

    # ------------------------------------------------------------------ plumbing
    def list_tools(self) -> list[dict]:
        w = {"type": "string", "description": "Wellbore name, e.g. 15/9-F-12"}
        d = {"type": "string", "description": "ISO date, inclusive"}
        per = {"well": w, "start": d, "end": d}
        return [
            {"name": "list_wells", "description": "Wells with production data and wells with logs.",
             "inputSchema": {"type": "object", "properties": {}}},
            {"name": "production_total", "description": "Sum of daily oil, gas or water volume (Sm3).",
             "inputSchema": {"type": "object", "properties": {**per, "quantity": {"enum": list(QUANTITIES)}},
                             "required": ["well", "start", "end", "quantity"]}},
            {"name": "shut_in_days", "description": "Days with zero on-stream hours.",
             "inputSchema": {"type": "object", "properties": per, "required": list(per)}},
            {"name": "water_cut", "description": "Water / (water + oil), by volume.",
             "inputSchema": {"type": "object", "properties": per, "required": list(per)}},
            {"name": "top_producer", "description": "Well with the largest total of a quantity.",
             "inputSchema": {"type": "object", "properties": {"start": d, "end": d, "quantity": {"enum": list(QUANTITIES)}},
                             "required": ["start", "end", "quantity"]}},
            {"name": "sensor_mean", "description": "Mean of a daily-average sensor over the period.",
             "inputSchema": {"type": "object", "properties": {**per, "sensor": {"enum": list(SENSOR_NAMES)}},
                             "required": ["well", "start", "end", "sensor"]}},
            {"name": "formation_top", "description": "Measured depth (m) of a formation pick.",
             "inputSchema": {"type": "object", "properties": {"well": w, "pick": {"type": "string"}},
                             "required": ["well", "pick"]}},
            {"name": "log_mean", "description": "Mean of a log curve between two measured depths.",
             "inputSchema": {"type": "object", "properties": {"well": w, "curve": {"enum": list(LOG_CURVES)},
                                                              "top_md": {"type": "number"}, "base_md": {"type": "number"}},
                             "required": ["well", "curve", "top_md", "base_md"]}},
        ]

    def call(self, name: str, **args) -> ToolResult:
        fn: Callable | None = getattr(self, f"_t_{name}", None)
        entry = {"seq": len(self.audit) + 1, "tool": name, "args": args}
        if fn is None:
            entry.update(ok=False, error="unknown tool")
            self.audit.append(entry)
            raise ToolError(f"unknown tool {name}")
        try:
            res = fn(**args)
        except ToolError as e:
            entry.update(ok=False, error=str(e))
            self.audit.append(entry)
            raise
        entry.update(ok=True, rows=res.rows, source=res.source, sha256=self.hashes.get(res.source, ""))
        self.audit.append(entry)
        return res

    # ------------------------------------------------------------------ helpers
    def _period(self, well: str | None, start: str, end: str) -> pd.DataFrame:
        try:
            a, b = pd.Timestamp(start), pd.Timestamp(end)
        except Exception as e:  # noqa: BLE001
            raise ToolError(f"bad date: {e}") from e
        if b < a:
            raise ToolError("end before start")
        df = self.production
        if well is not None:
            if well not in set(df["WELL"]):
                raise ToolError(f"unknown well {well}")
            df = df[df["WELL"] == well]
        return df[(df["DATE"] >= a) & (df["DATE"] <= b)]

    # ------------------------------------------------------------------ tools
    def _t_list_wells(self) -> ToolResult:
        v = {"production": sorted(self.production["WELL"].unique()), "logs": sorted(self.logs["WELL"].unique())}
        return ToolResult(v, "", len(v["production"]) + len(v["logs"]), self.PROD_FILE)

    def _t_production_total(self, well: str, start: str, end: str, quantity: str) -> ToolResult:
        if quantity not in QUANTITIES:
            raise ToolError(f"unknown quantity {quantity}")
        d = self._period(well, start, end)[QUANTITIES[quantity]].dropna()
        return ToolResult(float(d.sum()) if len(d) else None, "Sm3", len(d), self.PROD_FILE)

    def _t_shut_in_days(self, well: str, start: str, end: str) -> ToolResult:
        d = self._period(well, start, end)["ON_STREAM_HRS"].dropna()
        return ToolResult(int((d == 0).sum()) if len(d) else None, "days", len(d), self.PROD_FILE)

    def _t_water_cut(self, well: str, start: str, end: str) -> ToolResult:
        d = self._period(well, start, end)[["BORE_OIL_VOL", "BORE_WAT_VOL"]].dropna()
        liquid = float(d.sum().sum())
        if not len(d) or liquid <= 0:
            return ToolResult(None, "fraction", 0, self.PROD_FILE)
        return ToolResult(float(d["BORE_WAT_VOL"].sum() / liquid), "fraction", len(d), self.PROD_FILE)

    def _t_top_producer(self, start: str, end: str, quantity: str) -> ToolResult:
        if quantity not in QUANTITIES:
            raise ToolError(f"unknown quantity {quantity}")
        d = self._period(None, start, end).dropna(subset=[QUANTITIES[quantity]])
        tot = d.groupby("WELL")[QUANTITIES[quantity]].sum()
        if not len(d) or tot.max() <= 0:
            return ToolResult(None, "", 0, self.PROD_FILE)
        return ToolResult(str(tot.idxmax()), "", len(d), self.PROD_FILE)

    def _t_sensor_mean(self, well: str, start: str, end: str, sensor: str) -> ToolResult:
        if sensor not in SENSOR_NAMES:
            raise ToolError(f"unknown sensor {sensor}")
        col, unit = SENSOR_NAMES[sensor]
        d = self._period(well, start, end)[col].dropna()
        return ToolResult(float(d.mean()) if len(d) else None, unit, len(d), self.PROD_FILE)

    def _t_formation_top(self, well: str, pick: str) -> ToolResult:
        t = self.tops[self.tops["WELL"] == well]
        if not len(t):
            raise ToolError(f"unknown well {well}")
        m = t[t["PICKS"].str.lower() == pick.lower()]
        if not len(m):
            return ToolResult(None, "m MD", 0, self.TOPS_FILE)
        return ToolResult(float(m["MD"].iloc[0]), "m MD", len(m), self.TOPS_FILE)

    def _t_log_mean(self, well: str, curve: str, top_md: float, base_md: float) -> ToolResult:
        if curve not in LOG_CURVES:
            raise ToolError(f"unknown curve {curve}")
        g = self.logs[self.logs["WELL"] == well]
        if not len(g):
            raise ToolError(f"unknown well {well}")
        d = g[(g["DEPTH"] >= top_md) & (g["DEPTH"] <= base_md)][curve].dropna()
        src = well.replace("15/9-", "15_9-").replace(" ", "") + ".LAS"
        return ToolResult(float(np.mean(d)) if len(d) else None, "", len(d), src)
