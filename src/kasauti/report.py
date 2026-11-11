"""Run all tracks, write results.json, RESULTS.md and figures."""
from __future__ import annotations

import json
import math
import platform
from importlib import metadata
from pathlib import Path

from . import __version__, data, production, subsurface
from .agent import agents, scoring, tasks, tools

BLUE, ORANGE, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def environment() -> dict:
    libs = {}
    for lib in ("numpy", "pandas", "scikit-learn", "lasio", "openpyxl"):
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            libs[lib] = "missing"
    return {"kasauti": __version__, "python": platform.python_version(), **libs}


def run_all(data_dir: Path | None = None, agent=None, quiet: bool = False) -> dict:
    say = (lambda *a: None) if quiet else print
    hashes = data.verify(data_dir)
    say("data verified against manifest")
    prod, logs, tops = data.load_production(data_dir), data.load_logs(data_dir), data.load_tops(data_dir)
    say("track S: sonic synthesis, leave-one-well-out")
    s = subsurface.run(logs)
    say("track P: virtual metering and forecasting, walk-forward")
    p = production.run(prod)
    say("track A: agent tasks")
    task_list = tasks.build(prod, logs, tops)
    make = lambda: tools.Toolbox(prod, logs, tops, hashes)  # noqa: E731
    roster = [cls() for cls in agents.BUILTIN.values()] + ([agent] if agent else [])
    a = {"n_tasks": len(task_list), "agents": {}}
    for ag in roster:
        r = scoring.evaluate(ag, task_list, make)
        r.pop("tasks")
        a["agents"][ag.name] = r
    return _strip({"environment": environment(), "data_sha256": hashes,
                   "subsurface": s, "production": p, "agent": a})


def compare(new, old, path="", rel=1e-6) -> list[str]:
    """List numeric or structural differences between two result trees."""
    diffs = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k == "environment":
                continue
            if k not in old or k not in new:
                diffs.append(f"{path}/{k}: present in only one result")
            else:
                diffs += compare(new[k], old[k], f"{path}/{k}", rel)
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            diffs.append(f"{path}: length {len(new)} vs {len(old)}")
        else:
            for i, (a, b) in enumerate(zip(new, old)):
                diffs += compare(a, b, f"{path}[{i}]", rel)
    elif isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool):
        if not math.isclose(new, old, rel_tol=rel, abs_tol=1e-9):
            diffs.append(f"{path}: {new!r} vs published {old!r}")
    elif new != old:
        diffs.append(f"{path}: {new!r} vs published {old!r}")
    return diffs


# ------------------------------------------------------------------------- figures

def _ax(fig_w=7.2, fig_h=3.4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d2")
    ax.tick_params(length=0)
    return fig, ax, plt


def _save(fig, plt, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _paired(ax, labels, a, b, name_a, name_b, fmt):
    import numpy as np

    y = np.arange(len(labels))[::-1]
    ax.barh(y + 0.19, a, height=0.34, color=BLUE, label=name_a)
    ax.barh(y - 0.19, b, height=0.34, color=ORANGE, label=name_b)
    top = max(max(a), max(b))
    for yy, v in list(zip(y + 0.19, a)) + list(zip(y - 0.19, b)):
        ax.text(v + top * 0.012, yy, fmt(v), va="center", color=INK, fontsize=9)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, top * 1.14)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9, borderaxespad=0.2)


def figures(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    s, p, a = res["subsurface"]["targets"], res["production"], res["agent"]["agents"]

    fig, ax, plt = _ax(7.2, 2.6)
    v = p["vfm"]["leak_gap"]
    rows = [("Sonic DT", s["DT"]["leak_gap"]["mae_inflation"],
             f"MAE {s['DT']['leak_gap']['mae_random']:.1f} to {s['DT']['leak_gap']['mae_blind']:.1f} us/ft"),
            ("Shear DTS", s["DTS"]["leak_gap"]["mae_inflation"],
             f"MAE {s['DTS']['leak_gap']['mae_random']:.1f} to {s['DTS']['leak_gap']['mae_blind']:.1f} us/ft"),
            ("Daily oil rate", v["wape_inflation"], f"WAPE {v['wape_random']:.1%} to {v['wape_forward']:.1%}")]
    ys = list(range(len(rows)))[::-1]
    for y, (_, x, note) in zip(ys, rows):
        ax.barh(y, x, height=0.5, color=BLUE)
        ax.text(x + 0.05, y, f"{x:.1f}x   {note}", va="center", fontsize=9)
    ax.axvline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0, max(r[1] for r in rows) * 1.75)
    ax.set_title("How much worse the same model scores once the split is honest", loc="left", fontsize=11)
    ax.set_xlabel("Honest-split error as a multiple of shuffled-row error (dashed = no gap)")
    _save(fig, plt, out / "leak_gap.png")

    fig, ax, plt = _ax()
    folds = s["DT"]["folds"]
    _paired(ax, [f["test_well"] for f in folds],
            [100 * f["interval"]["naive_coverage"] for f in folds],
            [100 * f["interval"]["grouped_coverage"] for f in folds],
            "Calibrated on shuffled rows", "Calibrated on unseen wells", lambda x: f"{x:.0f}%")
    ax.axvline(90, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(0, 112)
    ax.set_title("Do 90% intervals hold on a well the model never saw? (sonic DT)", loc="left", fontsize=11, pad=24)
    ax.set_xlabel("Share of true values inside the interval (dashed = the 90% promised)")
    _save(fig, plt, out / "interval_coverage.png")

    fig, ax, plt = _ax(7.2, 3.0)
    names = {"last_calibration": "Last 30-day calibration (no model)", "ridge": "Ridge regression",
             "gbm": "Gradient boosting, all history", "gbm_recent": "Gradient boosting, last 365 days"}
    pooled = p["vfm"]["pooled"]
    ys = list(range(len(names)))[::-1]
    for y, k in zip(ys, names):
        w, (lo, hi) = 100 * pooled[k]["wape"], pooled[k]["wape_ci95"]
        ax.barh(y, w, height=0.5, color=BLUE)
        ax.plot([100 * lo, 100 * hi], [y, y], color=INK, lw=1.2)
        ax.text(100 * hi + 0.8, y, f"{w:.1f}%", va="center", fontsize=9)
    ax.set_yticks(ys, list(names.values()))
    ax.set_xlim(0, 100 * max(pooled[k]["wape_ci95"][1] for k in names) * 1.12)
    ax.set_title("Daily oil-rate estimation, walk-forward (line = 95% interval)", loc="left", fontsize=11)
    ax.set_xlabel("WAPE (lower is better)")
    _save(fig, plt, out / "virtual_metering.png")

    fig, ax, plt = _ax(7.2, 2.9)
    r = a["reference"]
    _paired(ax, ["All gated requests", "Red (safety-critical) only"],
            [100 * r["gate_recall"], 100 * r["red_recall"]],
            [100 * r["probe"]["gate_recall"], 100 * r["probe"]["red_recall"]],
            "Authored set (40 prompts)", "Held-out probe (30 prompts)", lambda x: f"{x:.0f}%")
    ax.set_xlim(0, 118)
    ax.set_title("Keyword safety critic: share of risky requests sent to a human", loc="left", fontsize=11, pad=24)
    ax.set_xlabel("Gate recall (higher is better)")
    _save(fig, plt, out / "safety_gate.png")


# ------------------------------------------------------------------------- markdown

def markdown(res: dict) -> str:
    s, p, a = res["subsurface"]["targets"], res["production"], res["agent"]
    L = ["# Kasauti results", "",
         "Generated by `kasauti run`. Do not edit by hand; `kasauti reproduce` checks every number below.", "",
         "Environment: " + ", ".join(f"{k} {v}" for k, v in res["environment"].items()), ""]

    L += ["## Track S: sonic-log synthesis (leave-one-well-out)", ""]
    for t in ("DT", "DTS"):
        r = s[t]
        L += [f"### {t} (us/ft), rows per well: " + ", ".join(f"{w} {n}" for w, n in r["rows_per_well"].items()), "",
              "| Method | MAE | RMSE | R2 pooled | R2 worst well |", "|---|---|---|---|---|"]
        label = {"train_mean": "Training mean", "gardner": "Gardner (1974) from density", "ridge": "Ridge regression", "gbm": "Gradient boosting"}
        for k, n in label.items():
            if k in r["pooled"]:
                m = r["pooled"][k]
                L.append(f"| {n} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} | {m['worst_well_r2']:.3f} |")
        g, i = r["leak_gap"], r["pooled"]["interval"]
        L += ["", f"Leak gap: the same gradient boosting scores MAE {g['mae_random']:.2f} on shuffled rows and "
                  f"{g['mae_blind']:.2f} on blind wells, {g['mae_inflation']:.2f} times worse.",
              f"Interval coverage at a nominal {i['nominal']:.0%}: {i['naive_coverage']:.1%} when calibrated on shuffled rows "
              f"(worst well {i['worst_well_naive_coverage']:.1%}); {i['grouped_coverage']:.1%} when calibrated on unseen wells "
              f"(worst well {i['worst_well_grouped_coverage']:.1%}).", ""]

    v = p["vfm"]
    L += ["## Track P1: virtual flow metering (walk-forward)", "",
          f"{v['n_scored_days']} scored well-days, {v['scored_period'][0]} to {v['scored_period'][1]}. {v['protocol'].capitalize()}.", "",
          "| Method | WAPE | 95% interval | Bias | R2 |", "|---|---|---|---|---|"]
    label = {"last_calibration": "Last 30-day calibration (no model)", "ridge": "Ridge regression",
             "gbm": "Gradient boosting, all history", "gbm_recent": "Gradient boosting, last 365 days"}
    for k, n in label.items():
        m = v["pooled"][k]
        L.append(f"| {n} | {m['wape']:.1%} | {m['wape_ci95'][0]:.1%} to {m['wape_ci95'][1]:.1%} | {m['bias']:+.1%} | {m['r2']:.3f} |")
    L += ["", "WAPE by well:", "", "| Well | Days | " + " | ".join(label.values()) + " |", "|---|---|" + "---|" * len(label)]
    for w, d in v["per_well"].items():
        L.append(f"| {w} | {d['n']} | " + " | ".join(f"{d[k]['wape']:.1%}" for k in label) + " |")
    g = v["leak_gap"]
    L += ["", f"Leak gap: gradient boosting scores WAPE {g['wape_random']:.1%} on shuffled days and {g['wape_forward']:.1%} "
              f"walking forward, {g['wape_inflation']:.1f} times worse.", ""]

    f = p["forecast"]
    L += ["## Track P2: oil forecast (rolling origin)", "", f["protocol"].capitalize() + ".", "",
          "| Horizon | Forecasts | Method | WAPE | Bias | Median APE | Skill vs 30-day persistence |", "|---|---|---|---|---|---|---|"]
    label = {"persistence_30d": "Persistence, last 30 days", "persistence_7d": "Persistence, last 7 days", "arps_exponential": "Arps exponential decline"}
    for h, d in f["horizons"].items():
        for k, n in label.items():
            m = d["methods"][k]
            L.append(f"| {h} days | {d['n_forecasts']} | {n} | {m['wape']:.1%} | {m['bias']:+.1%} | {m['median_ape']:.1%} | {m['skill_vs_persistence_30d']:+.1%} |")

    L += ["", "## Track A: agent tasks", "",
          f"{a['n_tasks']} tasks. The built-in agents contain no language model; see METHOD.md section 6 for what these rows do and do not show.", "",
          "| Agent | Answer accuracy | Abstention recall | Fabrication | Wrongful refusal | Gate recall | Red recall | False gate | Probe gate recall | Probe red recall | Probe false gate |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in a["agents"].items():
        pr = r["probe"]
        L.append(f"| {n} | {r['answer_accuracy']:.0%} | {r['abstention_recall']:.0%} | {r['fabrication_rate']:.0%} | "
                 f"{r['wrongful_abstention_rate']:.0%} | {r['gate_recall']:.0%} | {r['red_recall']:.0%} | {r['false_gate_rate']:.0%} | "
                 f"{pr['gate_recall']:.0%} | {pr['red_recall']:.0%} | {pr['false_gate_rate']:.0%} |")
    L += ["", "## Data", "", "| File | SHA-256 |", "|---|---|"]
    L += [f"| {k} | `{h}` |" for k, h in res["data_sha256"].items()]
    L += ["", "Volve data: Equinor and the former Volve licence partners. See DATA_LICENCE.md.", ""]
    return "\n".join(L)


def write(res: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    (out / "RESULTS.md").write_text(markdown(res))
    figures(res, out / "figures")
