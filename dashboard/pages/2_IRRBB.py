import sys
import os
import streamlit as st
from src import compute, queries
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(page_title="IRRBB", layout="wide")

with st.sidebar:
    scenario_label = st.selectbox(
        "Select Scenario",
        ["Baseline", "Stress: ECB Stress", "Stress: Liquidity Shock"],
        key="irrbb_scenario"
    )

scenario_map = {
    "Baseline": None,
    "Stress: ECB Stress": 1,
    "Stress: Liquidity Shock": 2
}

scenario_id = scenario_map[scenario_label]

# ==========================================================
# Risk Tiles
# ==========================================================

st.markdown("### Risk Summary Tile")

summary = compute.calculate_irrbb_risk_summary(
    shock_bps_list=[-300, -200, -100, 0, 100, 200, 300],
    scenario_id=scenario_id
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total PV01", f"{summary['Total PV01']:,.2f} EUR")
col2.metric("Max ∆EVE", f"{summary['Max ∆EVE']:,.2f} EUR")
col3.metric("Max ∆NII", f"{summary['Max ∆NII']:,.2f} EUR")
col4.metric(
    label="∆EVE / Tier1 (%)",
    value=f"{summary['∆EVE Ratio'] * 100:.2f}%",
    delta="✅ OK" if summary['∆EVE Ratio'] < 0.15 else "❌ Breach",
    delta_color="normal" if summary['∆EVE Ratio'] < 0.15 else "inverse"
)

# ==========================================================
# PV01 Exposure by Tenor Bucket
# ==========================================================

st.subheader("IRRBB PV01 Exposure by Tenor Bucket")

pv01_df = compute.calculate_pv01_profile()

fig = px.bar(
    pv01_df,
    x='tenor_bucket',
    y='pv01',
    color='tenor_bucket',
    labels={'pv01': 'PV01 (EUR)', 'tenor_bucket': 'Maturity Bucket'},
    title="PV01 Exposure by Tenor Bucket"
)

fig.update_layout(
    showlegend=False,
    yaxis_tickformat=',.0f',
    height=400
)

fig.update_traces(hovertemplate='Bucket: %{x}<br>PV01: %{y:,.4f} EUR')

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ∆EVE Under EBA IRRBB Shock Scenarios
# ==========================================================

st.subheader("∆EVE Under EBA IRRBB Shock Scenarios")

df_eve = compute.calculate_eve_eba_scenarios(scenario_id=scenario_id)

fig = px.bar(
    df_eve,
    x='Scenario',
    y='Delta EVE',
    text_auto='.2f',
    color='Delta EVE',
    color_continuous_scale='RdYlGn',
    title=f"∆EVE Across EBA IRRBB Shocks – Scenario: {scenario_label}"
)

fig.update_layout(
    yaxis_title="Delta EVE (EUR)",
    xaxis_title="Scenario",
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# ∆NII – Net Interest Income under EBA Shocks
# ==========================================================

st.subheader("∆NII – Net Interest Income under EBA Shocks")

# Run ∆NII calculation
df_nii = compute.calculate_nii_eba_scenarios(scenario_id=scenario_id)

# Bar chart
fig = px.bar(
    df_nii,
    x="Scenario",
    y="Delta NII",
    title="∆NII Under EBA IRRBB Shocks",
    color="Scenario",
    labels={"Delta NII": "∆NII (EUR)"},
    text_auto=".2s"
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Parallel Shock ∆EVE Sensitivity
# ==========================================================

st.subheader("Parallel Shock ∆EVE Sensitivity")

shock_bps = st.slider(
    "Select Parallel Interest Rate Shock (bps)",
    min_value=-300,
    max_value=300,
    value=0,
    step=25,
    key="eve_shock_slider"
)

sensitivity = compute.calculate_eve_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)

st.metric(label="Shock (bps)", value=sensitivity['Shock (bps)'])
st.metric(label="Total PV01", value=f"{sensitivity['Total PV01']:,.2f} EUR")
st.metric(label="∆EVE", value=f"{sensitivity['Delta EVE']:,.2f} EUR")

# ==========================================================
# Interactive Yield Curve Slider
# ==========================================================

st.subheader("Interactive Yield Curve Shift Explorer")

# --- Define buckets and base curve ---
buckets = ['0-1y', '1-3y', '3-5y', '5-10y', '10y+']
baseline_curve = [0.01, 0.0125, 0.015, 0.0175, 0.02]  # 1% to 2%

# --- EBA Preset Shocks (in bps) ---
eba_presets = {
    "Parallel Up":      [200, 200, 200, 200, 200],
    "Parallel Down":    [-200, -200, -200, -200, -200],
    "Steepener":        [-50, 0, 100, 150, 200],
    "Flattener":        [250, 200, 150, 100, 50],
    "Short Rate Up":    [300, 200, 100, 0, 0],
    "Short Rate Down":  [-300, -200, -100, 0, 0],
    "Reset":            [0, 0, 0, 0, 0]
}

# 👇 Add scenario buttons
st.markdown("Choose EBA Scenario or Manual Shift:")
selected_eba = st.radio(
    "Preset Scenario:",
    list(eba_presets.keys()),
    index=6,
    horizontal=True,
    key="eba_radio"
)

# Load preset if not manual
if selected_eba != "Reset":
    custom_shocks_bps = eba_presets[selected_eba]
else:
    custom_shocks_bps = [0] * len(buckets)

# --- Manual Shocks Slider ---
# 👇 Display sliders with preset values
st.markdown("Adjust the yield curve manually (bps):")
cols = st.columns(len(buckets))
for i, b in enumerate(buckets):
    with cols[i]:
        custom_shocks_bps[i] = st.slider(
            label=b,
            min_value=-300,
            max_value=300,
            value=custom_shocks_bps[i],
            step=25,
            key=f"shock_slider_{b}"
        )

# --- Apply shock in decimal form ---
custom_shocks = [s / 10_000 for s in custom_shocks_bps]
shifted_curve = [y + s for y, s in zip(baseline_curve, custom_shocks)]

# --- Plot Curves ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=buckets, y=baseline_curve, name="Baseline", line=dict(color='blue')))
fig.add_trace(go.Scatter(x=buckets, y=shifted_curve, name="Shifted Curve", line=dict(color='orange')))
fig.update_layout(title="Yield Curve Shift", yaxis_title="Yield", xaxis_title="Tenor Bucket")
st.plotly_chart(fig, use_container_width=True)

# --- Recalculate ∆EVE and ∆NII ---
def calculate_curve_shift_impact(shocks, scenario_id=None):
    irrbb = queries.get_irrbb(scenario_id)
    cashflows = queries.get_cashflows(scenario_id)

    # PV01
    pv01_by_bucket = irrbb.groupby('tenor_bucket')['pv01'].sum().reindex(buckets).fillna(0)
    delta_eve = (pv01_by_bucket * shocks).sum()

    # Repricing Gap
    cashflows['date'] = pd.to_datetime(cashflows['date'])
    cashflows['maturity_date'] = pd.to_datetime(cashflows['maturity_date'])
    cashflows['maturity_days'] = (cashflows['maturity_date'] - cashflows['date']).dt.days

    def assign_bucket(days):
        if days <= 7:
            return '0-1y'
        elif days <= 30:
            return '1-3y'
        elif days <= 90:
            return '3-5y'
        elif days <= 180:
            return '5-10y'
        else:
            return '10y+'

    cashflows['bucket'] = cashflows['maturity_days'].apply(assign_bucket)
    cashflows['signed_amount'] = cashflows.apply(
        lambda row: row['amount'] if row['direction'] == 'inflow' else -row['amount'], axis=1
    )
    gap_by_bucket = cashflows.groupby('bucket')['signed_amount'].sum().reindex(buckets).fillna(0)
    delta_nii = (gap_by_bucket * shocks).sum()

    return delta_eve, delta_nii

# --- Results ---
delta_eve, delta_nii = calculate_curve_shift_impact(custom_shocks)
col1, col2 = st.columns(2)
col1.metric("∆EVE", f"{delta_eve:,.2f} EUR")
col2.metric("∆NII", f"{delta_nii:,.2f} EUR")

