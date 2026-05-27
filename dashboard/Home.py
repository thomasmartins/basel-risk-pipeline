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
# PV01 by Tenor Bucket Chart
# ===========================================================
st.subheader("PV01 Profile by Tenor Bucket")
st.bar_chart(
    data=pv01.set_index("tenor_bucket")["pv01"], width='stretch'
)

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

with st.expander("🔍 Show Raw IRRBB Data"):
    st.dataframe(queries.get_irrbb(scenario_id=scenario_id))

with st.expander("🔍 Show Raw Balance Sheet Data"):
    st.dataframe(queries.get_balance_sheet(scenario_id=scenario_id))
