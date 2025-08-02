import streamlit as st
import pandas as pd
import plotly.express as px
from src import compute, queries
import sys
import os


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(page_title="Stress Testing Panel", layout="wide")

st.title("Stress Testing Panel")

# --- Sidebar Inputs ---
st.sidebar.header("Stress Test Parameters")

shock_bps = st.sidebar.slider("Interest Rate Shock (bps)", -300, 300, 200, step=25)
retail_withdrawal_pct = st.sidebar.slider("Retail Withdrawal (%)", 0.0, 1.0, 0.2, step=0.05)
wholesale_withdrawal_pct = st.sidebar.slider("Wholesale Withdrawal (%)", 0.0, 1.0, 0.4, step=0.05)
rwa_stress_pct = st.sidebar.slider("RWA Increase (%)", 0.0, 1.0, 0.1, step=0.05)

# --- Compute ---
results = compute.run_stress_test(
    shock_bps=shock_bps,
    retail_withdrawal_pct=retail_withdrawal_pct,
    wholesale_withdrawal_pct=wholesale_withdrawal_pct,
    rwa_stress_pct=rwa_stress_pct
)

# --- Metrics ---
st.subheader("Key Risk Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("LCR", f"{results['LCR (Stressed)']:.2f}", f"{results['LCR (Stressed)'] - results['LCR (Base)']:+.2f}")
col2.metric("NSFR", f"{results['NSFR (Stressed)']:.2f}", f"{results['NSFR (Stressed)'] - results['NSFR (Base)']:+.2f}")
col3.metric("∆EVE", f"{results['∆EVE (Stressed)']:,.2f} EUR")

col4, col5, col6 = st.columns(3)
col4.metric("CET1 Ratio", f"{results['CET1 Ratio (Stressed)']:.2%}", f"{results['CET1 Ratio (Stressed)'] - results['CET1 Ratio (Base)']:+.2%}")
col5.metric("Tier1 Ratio", f"{results['Tier1 Ratio (Stressed)']:.2%}", f"{results['Tier1 Ratio (Stressed)'] - results['Tier1 Ratio (Base)']:+.2%}")
col6.metric("∆NII", f"{results['∆NII (Stressed)']:,.2f} EUR")

# --- Comparison Chart ---
st.subheader("Before vs. After Stress")

chart_data = pd.DataFrame({
    "Metric": [
        "LCR", "NSFR", "CET1 Ratio", "Tier1 Ratio", "∆EVE", "∆NII"
    ],
    "Base": [
        results['LCR (Base)'],
        results['NSFR (Base)'],
        results['CET1 Ratio (Base)'],
        results['Tier1 Ratio (Base)'],
        results['∆EVE (Base)'],
        results['∆NII (Base)'],
    ],
    "Stressed": [
        results['LCR (Stressed)'],
        results['NSFR (Stressed)'],
        results['CET1 Ratio (Stressed)'],
        results['Tier1 Ratio (Stressed)'],
        results['∆EVE (Stressed)'],
        results['∆NII (Stressed)'],
    ]
})

chart_data = chart_data.melt(id_vars="Metric", var_name="Condition", value_name="Value")

# Split data
percent_metrics = ["LCR", "NSFR", "CET1 Ratio", "Tier1 Ratio"]
absolute_metrics = ["∆EVE", "∆NII"]

# --- Percent Metrics ---
chart_percent = chart_data[chart_data['Metric'].isin(percent_metrics)]
# Metrics that are in decimal form and need to be multiplied by 100
percent_metrics_needing_scaling = ["NSFR", "CET1 Ratio", "Tier1 Ratio"]

# Apply scaling
chart_percent['Value'] = chart_percent.apply(
    lambda row: row['Value'] * 100 if row['Metric'] in percent_metrics_needing_scaling else row['Value'],
    axis=1
)

fig1 = px.bar(
    chart_percent,
    x="Metric",
    y="Value",
    color="Condition",
    barmode="group",
    text_auto='.2f',
    title="Regulatory Ratios: Before vs. After"
)
fig1.update_yaxes(title="%")

st.plotly_chart(fig1, use_container_width=True)