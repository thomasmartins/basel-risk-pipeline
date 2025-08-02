import streamlit as st
import pandas as pd
import sys
import os
import plotly.graph_objects as go
import datetime

repo_root = os.path.dirname(os.path.abspath(__file__))  # dashboard/
repo_root = os.path.abspath(os.path.join(repo_root, ".."))  # up one level

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from src import compute, queries

st.set_page_config(page_title="Basel III Risk Dashboard", layout="wide")

# ===========================================================
# Sidebar - Scenario Selector
# ===========================================================
st.sidebar.title("Scenario Selector")

scenarios = queries.get_scenarios()
scenario_map = dict(zip(scenarios['name'], scenarios['id']))

scenario_choice = st.sidebar.selectbox(
    "Select Scenario",
    options=scenario_map.keys(),
    index=0
)

scenario_id = scenario_map[scenario_choice]

# ===========================================================
# Dashboard Title
# ===========================================================

st.title("Basel III Risk Dashboard")
st.markdown("""
Welcome to the Basel III Risk Dashboard.  
Use the sidebar to navigate to:
- Liquidity Risk
- Capital Adequacy & RWA
- Interest Rate Risk (IRRBB)
""")

# ===========================================================
# KPI Tiles
# ===========================================================

st.subheader("Main KPIs")
st.subheader(f"Scenario: {scenario_choice}")

lcr = compute.calculate_lcr(scenario_id)
nsfr = compute.calculate_nsfr(scenario_id)
capital = compute.calculate_capital_ratios(scenario_id)
eve = compute.calculate_eve_sensitivity(scenario_id=scenario_id)
pv01 = compute.calculate_pv01_profile(scenario_id)
total_pv01 = pv01['pv01'].sum()



def get_kpi_statuses(lcr, nsfr, capital, eve, pv01):
    def light(val, green, yellow, higher=True):
        if higher:
            return "🟢" if val >= green else "🟡" if val >= yellow else "🔴"
        else:
            return "🟢" if val <= green else "🟡" if val <= yellow else "🔴"

    tier1_cap = capital['Tier1 Ratio'] * capital['RWA']
    eve_ratio = eve['Delta EVE'] / tier1_cap if tier1_cap > 0 else 0

    return {
        "LCR": {
            "value": lcr['LCR'],
            "label": f"{light(lcr['LCR'], 1.0, 0.9)} {lcr['LCR']:.2f}"
        },
        "NSFR": {
            "value": nsfr['NSFR'],
            "label": f"{light(nsfr['NSFR'], 1.0, 0.9)} {nsfr['NSFR']:.2f}"
        },
        "CET1 Ratio": {
            "value": capital['CET1 Ratio'],
            "label": f"{light(capital['CET1 Ratio'], 0.07, 0.045)} {capital['CET1 Ratio']:.2%}"
        },
        "Total Capital Ratio": {
            "value": capital['Total Capital Ratio'],
            "label": f"{light(capital['Total Capital Ratio'], 0.10, 0.08)} {capital['Total Capital Ratio']:.2%}"
        },
        "Total PV01": {
            "value": pv01['Total PV01'],
            "label": f"{pv01['Total PV01']:.2f}"
        },
        "Delta EVE (+200bps)": {
            "value": eve['Delta EVE'],
            "label": f"{light(eve_ratio, 0.15, 0.15, higher=False)} {eve['Delta EVE']:.2f}"
        }
    }

kpis = get_kpi_statuses(lcr, nsfr, capital, eve, {"Total PV01": total_pv01})

# Display
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("LCR", kpis["LCR"]["label"])
kpi2.metric("NSFR", kpis["NSFR"]["label"])
kpi3.metric("CET1 Ratio", kpis["CET1 Ratio"]["label"])
kpi4.metric("Total Capital Ratio", kpis["Total Capital Ratio"]["label"])

kpi5, kpi6 = st.columns(2)
kpi5.metric("Total PV01", kpis["Total PV01"]["label"])
kpi6.metric("Delta EVE (+200bps)", kpis["Delta EVE (+200bps)"]["label"])

# timestamp

# Hardcoded placeholder (replace with real metadata query later)
last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Display
st.markdown(f"🕒 **Last Data Refresh:** `{last_updated}`")


# ===========================================================
# PV01 by Tenor Bucket Chart
# ===========================================================
st.subheader("PV01 Profile by Tenor Bucket")

st.bar_chart(
    data=pv01.set_index('tenor_bucket')['pv01'],
    use_container_width=True
)

# Pillar 1 collapsable box

with st.expander("📘 What is Basel III Pillar 1?"):
    st.markdown("""
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
""")


# ===========================================================
# Data Inspectors (Optional MVP)
# ===========================================================
with st.expander("🔍 Show Raw Cashflows Data"):
    st.dataframe(queries.get_cashflows(scenario_id=scenario_id))

with st.expander("🔍 Show Raw RWA Data"):
    st.dataframe(queries.get_rwa(scenario_id=scenario_id))

with st.expander("🔍 Show Raw IRRBB Data"):
    st.dataframe(queries.get_irrbb(scenario_id=scenario_id))

with st.expander("🔍 Show Raw Balance Sheet Data"):
    st.dataframe(queries.get_balance_sheet(scenario_id=scenario_id))



