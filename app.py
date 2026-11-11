"""Kasauti in Streamlit: published results, plus a live agent console on the real data.

    pip install -e ".[app]"
    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Works whether or not the package has been pip-installed.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kasauti import data  # noqa: E402
from kasauti.agent import agents, tasks, tools  # noqa: E402

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
ATTRIBUTION = ("Contains data from the Volve field dataset, disclosed by Equinor ASA and the former "
               "Volve licence partners under the Equinor Open Data Licence.")

st.set_page_config(page_title="Kasauti", layout="wide")
st.title("Kasauti")
st.caption("A touchstone for operational AI, struck on real Volve field data.")

results_file = ROOT / "results" / "results.json"
if not results_file.exists():
    st.error("No results found. Run `kasauti run` in this folder first.")
    st.stop()
res = json.loads(results_file.read_text())
S, P, A = res["subsurface"]["targets"], res["production"], res["agent"]["agents"]
FIG = ROOT / "results" / "figures"


@st.cache_resource(show_spinner="Loading Volve data...")
def load():
    hashes = data.verify(RAW)
    return hashes, data.load_production(RAW), data.load_logs(RAW), data.load_tops(RAW)


tab_sum, tab_s, tab_p, tab_a, tab_try = st.tabs(
    ["Summary", "Subsurface", "Production", "Agent", "Try the agent"])

with tab_sum:
    v = P["vfm"]
    c = st.columns(4)
    c[0].metric("Leak gap, oil rate", f"{v['leak_gap']['wape_inflation']:.1f}x",
                help="Error walking forward in time, as a multiple of error on shuffled days")
    c[1].metric("90% intervals on a blind well", f"{S['DT']['pooled']['interval']['naive_coverage']:.0%}",
                help="Actual coverage with the usual shuffled-row calibration (sonic DT)")
    c[2].metric("Best model, oil-rate error", f"{v['pooled']['gbm_recent']['wape']:.1%}",
                delta=f"{v['pooled']['gbm_recent']['wape'] - v['pooled']['last_calibration']['wape']:+.1%} vs no model",
                delta_color="inverse")
    c[3].metric("Safety-critical requests gated", f"{A['reference']['probe']['red_recall']:.0%}",
                help="Keyword critic on 30 prompts written after it was frozen")
    left, right = st.columns(2)
    left.image(str(FIG / "leak_gap.png"))
    right.image(str(FIG / "safety_gate.png"))
    st.info("Every number here is read from results/results.json. `kasauti reproduce` re-computes and checks them.")

with tab_s:
    st.subheader("Sonic-log synthesis, scored on a well the model never saw")
    names = {"train_mean": "Training mean", "gardner": "Gardner (1974)", "ridge": "Ridge regression", "gbm": "Gradient boosting"}
    for t in ("DT", "DTS"):
        rows = [{"Method": n, "MAE (us/ft)": S[t]["pooled"][k]["mae"], "R2 pooled": S[t]["pooled"][k]["r2"],
                 "R2 worst well": S[t]["pooled"][k]["worst_well_r2"]} for k, n in names.items() if k in S[t]["pooled"]]
        st.markdown(f"**{t}**")
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, use_container_width=True)
    st.image(str(FIG / "interval_coverage.png"), width=760)

with tab_p:
    st.subheader("Daily oil rate, trained on the past and scored on the future")
    names = {"last_calibration": "Last 30-day calibration (no model)", "ridge": "Ridge regression",
             "gbm": "Gradient boosting, all history", "gbm_recent": "Gradient boosting, last 365 days"}
    v = P["vfm"]
    st.caption(f"{v['n_scored_days']:,} scored well-days, {v['scored_period'][0]} to {v['scored_period'][1]}")
    st.dataframe(pd.DataFrame([{"Method": n, "WAPE %": 100 * v["pooled"][k]["wape"],
                                "95% low": 100 * v["pooled"][k]["wape_ci95"][0], "95% high": 100 * v["pooled"][k]["wape_ci95"][1],
                                "Bias %": 100 * v["pooled"][k]["bias"]} for k, n in names.items()]).round(1),
                 hide_index=True, use_container_width=True)
    st.markdown("**Error (WAPE %) by well**")
    st.dataframe(pd.DataFrame([{"Well": w, "Days": d["n"], **{n: round(100 * d[k]["wape"], 1) for k, n in names.items()}}
                               for w, d in v["per_well"].items()]), hide_index=True, use_container_width=True)
    st.subheader("Oil forecast, rolling origin")
    fn = {"persistence_30d": "Persistence, 30 days", "persistence_7d": "Persistence, 7 days", "arps_exponential": "Arps exponential"}
    st.dataframe(pd.DataFrame([{"Horizon (days)": int(h), "Method": n, "WAPE %": 100 * d["methods"][k]["wape"],
                                "Bias %": 100 * d["methods"][k]["bias"],
                                "Skill vs 30-day persistence %": 100 * d["methods"][k]["skill_vs_persistence_30d"]}
                               for h, d in P["forecast"]["horizons"].items() for k, n in fn.items()]).round(1),
                 hide_index=True, use_container_width=True)

with tab_a:
    st.subheader("Agent tasks")
    st.caption("These agents contain no language model. They show the tasks are solvable and the scorer "
               "tells a safe agent from a useful one. See METHOD.md, section 6.")
    cols = {"answer_accuracy": "Answers correct", "abstention_recall": "Abstains correctly", "fabrication_rate": "Fabricates",
            "gate_recall": "Gate recall", "red_recall": "Red recall", "false_gate_rate": "False gate"}
    rows = []
    for name, r in A.items():
        row = {"Agent": name, **{lab: f"{r[k]:.0%}" for k, lab in cols.items()}}
        row.update({"Probe gate recall": f"{r['probe']['gate_recall']:.0%}", "Probe red recall": f"{r['probe']['red_recall']:.0%}"})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with tab_try:
    st.subheader("Ask the reference agent")
    try:
        hashes, prod, logs, tops = load()
    except data.DataError as e:
        st.warning(f"{e}  Run `kasauti fetch` in this folder, then reload.")
        st.stop()
    task_list = tasks.build(prod, logs, tops)
    examples = [t.question for t in task_list if t.id in ("F001", "F005", "F006", "U001", "S001", "S011", "X002")]
    pick = st.selectbox("Example questions", ["(type your own below)"] + examples)
    q = st.text_input("Question", value="" if pick.startswith("(") else pick)
    st.caption("The agent is rule-based and understands only the benchmark's question forms. "
               "The last example shows a safety-critical request the keyword critic misses.")
    if q:
        tb = tools.Toolbox(prod, logs, tops, hashes)
        r = agents.ReferenceAgent().answer(q, tb)
        badge = {"green": "🟢", "yellow": "🟡", "orange": "🟠", "red": "🔴"}[r.risk]
        st.markdown(f"**Risk:** {badge} {r.risk}")
        if r.requires_human:
            st.error(r.text)
        elif r.abstained:
            st.warning(r.text)
        else:
            st.success(r.text)
        if tb.audit:
            st.markdown("**Audit trail**")
            st.json(tb.audit)

st.divider()
st.caption(ATTRIBUTION + " Kasauti is independent work and is not endorsed by Equinor or its partners.")
