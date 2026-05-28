import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import compute, queries
from src.lineage import render_model_lineage
from src.scenario import scenario_sidebar

st.set_page_config(page_title="Basel III Risk Dashboard", layout="wide")

scenario_id = scenario_sidebar()
scenario_name = st.session_state.get("scenario_choice", "Baseline")

# ===========================================================
# Dashboard Title
# ===========================================================
st.title("Basel III Risk Dashboard")

render_model_lineage(expanded=False)
st.markdown(
    """
Welcome to the Basel III Risk Dashboard.
Use the sidebar to navigate to:
- Liquidity Risk
- Capital Adequacy & RWA
- Interest Rate Risk (IRRBB)
- Stress Testing
"""
)

# ===========================================================
# KPI Tiles
# ===========================================================
st.subheader("Main KPIs")
st.subheader(f"Scenario: {scenario_name}")

lcr = compute.calculate_lcr(scenario_id)
nsfr = compute.calculate_nsfr(scenario_id)
capital = compute.calculate_capital_ratios(scenario_id)
eve = compute.calculate_eve_sensitivity(scenario_id=scenario_id)
pv01 = compute.calculate_pv01_profile(scenario_id)
total_pv01 = pv01["pv01"].sum()


def get_kpi_statuses(lcr, nsfr, capital, eve, total_pv01):
    def light(val, green, yellow, higher=True):
        if higher:
            return "🟢" if val >= green else "🟡" if val >= yellow else "🔴"
        return "🟢" if val <= green else "🟡" if val <= yellow else "🔴"

    tier1_cap = capital["Tier1 Ratio"] * capital["RWA"]
    eve_ratio = abs(eve["Delta EVE"]) / tier1_cap if tier1_cap > 0 else 0

    return {
        "LCR": {"label": f"{light(lcr['LCR'], 1.0, 0.9)} {lcr['LCR']:.2f}"},
        "NSFR": {"label": f"{light(nsfr['NSFR'], 1.0, 0.9)} {nsfr['NSFR']:.2f}"},
        "CET1 Ratio": {
            "label": f"{light(capital['CET1 Ratio'], 0.07, 0.045)} {capital['CET1 Ratio']:.2%}"
        },
        "Total Capital Ratio": {
            "label": f"{light(capital['Total Capital Ratio'], 0.10, 0.08)} {capital['Total Capital Ratio']:.2%}"
        },
        "Total PV01": {"label": f"{total_pv01:.2f}"},
        "Delta EVE (+200bps)": {
            "label": f"{light(eve_ratio, 0.15, 0.15, higher=False)} {eve['Delta EVE']:.2f}"
        },
    }


kpis = get_kpi_statuses(lcr, nsfr, capital, eve, total_pv01)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("LCR", kpis["LCR"]["label"])
kpi2.metric("NSFR", kpis["NSFR"]["label"])
kpi3.metric("CET1 Ratio", kpis["CET1 Ratio"]["label"])
kpi4.metric("Total Capital Ratio", kpis["Total Capital Ratio"]["label"])

kpi5, kpi6 = st.columns(2)
kpi5.metric("Total PV01", kpis["Total PV01"]["label"])
kpi6.metric("Delta EVE (+200bps)", kpis["Delta EVE (+200bps)"]["label"])

# ===========================================================
# Cross-scenario KPI comparison
# ===========================================================
st.subheader("Headline ratios across scenarios")
st.caption(
    "Each ratio is expressed as a multiple of its regulatory minimum, so LCR/NSFR "
    "(threshold 1.00) and capital ratios (CET1 4.5%, Tier1 6%, Total 8%) sit on the "
    "same axis. A bar at 1.0 = exactly at the floor; below 1.0 = breach."
)

_KPI_DEFS = [
    ("LCR",                lambda sid: compute.calculate_lcr(sid)["LCR"],                  1.00, "{:.2f}"),
    ("NSFR",               lambda sid: compute.calculate_nsfr(sid)["NSFR"],                1.00, "{:.2f}"),
    ("CET1 Ratio",         lambda sid: compute.calculate_capital_ratios(sid)["CET1 Ratio"],          0.045, "{:.2%}"),
    ("Tier1 Ratio",        lambda sid: compute.calculate_capital_ratios(sid)["Tier1 Ratio"],         0.06,  "{:.2%}"),
    ("Total Capital Ratio",lambda sid: compute.calculate_capital_ratios(sid)["Total Capital Ratio"], 0.08,  "{:.2%}"),
]

_scenarios_df = queries.get_scenarios()
rows = []
for _, srow in _scenarios_df.iterrows():
    sid = int(srow["id"])
    sname = srow["name"]
    for kpi, fn, floor, fmt in _KPI_DEFS:
        try:
            val = float(fn(sid))
        except Exception:
            val = float("nan")
        rows.append({
            "scenario": sname,
            "kpi": kpi,
            "value": val,
            "floor": floor,
            "ratio_to_floor": val / floor if floor and val == val else float("nan"),
            "display": fmt.format(val),
        })
_compare_df = pd.DataFrame(rows)

_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
_fig = go.Figure()
for i, (kpi, _, _, _) in enumerate(_KPI_DEFS):
    sub = _compare_df[_compare_df["kpi"] == kpi]
    _fig.add_trace(go.Bar(
        x=sub["scenario"],
        y=sub["ratio_to_floor"],
        name=kpi,
        marker_color=_palette[i % len(_palette)],
        customdata=list(zip(sub["display"], sub["ratio_to_floor"])),
        hovertemplate="%{x}<br>" + kpi + ": %{customdata[0]} (×%{customdata[1]:.2f} floor)<extra></extra>",
    ))
_fig.add_hline(
    y=1.0, line_dash="dot", line_color="red",
    annotation_text="Regulatory floor (×1)", annotation_position="top right",
)
_fig.update_layout(
    barmode="group",
    xaxis_title="Scenario",
    yaxis_title="Value ÷ regulatory floor",
    yaxis=dict(rangemode="tozero"),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    height=420,
)
st.plotly_chart(_fig, width='stretch')

# Pillar 1 collapsable box
with st.expander("📘 What is Basel III Pillar 1?"):
    st.markdown(
        """
**Basel III Pillar 1** sets **minimum capital and liquidity requirements** for banks to ensure financial stability. This dashboard covers its four key components:

- **🧊 LCR (Liquidity Coverage Ratio):**
  Requires banks to hold enough **High-Quality Liquid Assets (HQLA)** to cover **30-day net cash outflows**. Minimum: **100%**.

- **🌊 NSFR (Net Stable Funding Ratio):**
  Ensures stable funding over a **1-year horizon**, matching asset/liability profiles. Minimum: **100%**.

- **📉 IRRBB (Interest Rate Risk in the Banking Book):**
  Banks must assess how **rate shocks** affect:
  - 🔸 **EVE (Economic Value of Equity)**
  - 🔸 **NII (Net Interest Income)**
  ∆EVE must not exceed **15% of Tier 1 capital**.

- **📊 RWA & Capital Adequacy:**
  Capital must cover **risk-weighted assets** (RWA) with:
  - CET1 ≥ **4.5%**
  - Tier 1 ≥ **6.0%**
  - Total Capital ≥ **8.0%**
  + **2.5% conservation buffer** under Pillar 1

---
ℹ️ This dashboard applies **EBA (European Banking Authority)** standards where relevant.
"""
    )

# ===========================================================
# Data Inspectors
# ===========================================================
with st.expander("🔍 Show Raw Cashflows Data"):
    st.dataframe(queries.get_cashflows(scenario_id=scenario_id))

with st.expander("🔍 Show Raw RWA Data"):
    st.dataframe(queries.get_rwa(scenario_id=scenario_id))

with st.expander("🔍 Show Raw Balance Sheet Data"):
    st.dataframe(queries.get_balance_sheet(scenario_id=scenario_id))
