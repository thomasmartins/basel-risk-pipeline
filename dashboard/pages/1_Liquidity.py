import sys
import os
import streamlit as st
from src import compute, queries
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


st.set_page_config(page_title="Liquidity Risk", layout="wide")

st.title("Liquidity Risk")

# Scenario selector
scenarios = queries.get_scenarios()
scenario_map = dict(zip(scenarios['name'], scenarios['id']))
scenario_choice = st.sidebar.selectbox("Select Scenario", options=scenario_map.keys(), index=0)
scenario_id = scenario_map[scenario_choice]

# KPIs
lcr = compute.calculate_lcr(scenario_id)
nsfr = compute.calculate_nsfr(scenario_id)

st.subheader(f"Scenario: {scenario_choice}")

k1, k2 = st.columns(2)
k1.metric("LCR", f"{lcr['LCR']:.2f}")
k2.metric("NSFR", f"{nsfr['NSFR']:.2f}")

# ==========================================================
# Waterfall Chart
# ==========================================================
st.subheader("LCR Waterfall")

h = lcr['HQLA']
out = -lcr['Outflows']
cap_in = min(lcr['Inflows'], lcr['Outflows'] * 0.75)
net_out = lcr['NetOutflows']

fig = go.Figure(go.Waterfall(
    name="LCR",
    orientation="v",
    measure=["absolute", "relative", "relative", "total"],
    x=["HQLA", "Outflows", "Inflows (capped)", "Net Outflows"],
    y=[h, out, cap_in, net_out],
    textposition="outside",
    text=[f"{h:,.0f}", f"{out:,.0f}", f"{cap_in:,.0f}", f"{net_out:,.0f}"],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig.update_layout(
    title="LCR Waterfall Breakdown",
    yaxis_title="EUR",
    waterfallgap=0.3
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# HQLA Composition
# ==========================================================

def hqla_to_waffle(hqla_dict):
    total = sum(hqla_dict.values())
    proportions = {k: round((v / total) * 100) for k, v in hqla_dict.items()}

    grid = []
    for i, (category, count) in enumerate(proportions.items()):
        grid.extend([category] * count)

    return grid

def get_hqla_treemap_data(scenario_id=None):
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()

    haircut_map = {
        'Level1': 0.0,
        'Level2A': float(params.get('haircut_level2a', 0.15)),
        'Level2B': float(params.get('haircut_level2b', 0.5)),
        'None': 1.0
    }

    # Filter only HQLA-eligible assets
    hqla = cashflows[cashflows['hqlatype'].isin(['Level1', 'Level2A', 'Level2B'])].copy()

    # Add both pre and post haircut columns
    hqla['Pre-Haircut'] = hqla['amount']
    hqla['Post-Haircut'] = hqla.apply(
        lambda x: x['amount'] * (1 - haircut_map.get(x['hqlatype'], 1)), axis=1
    )

    # Group and sum both columns
    grouped = hqla.groupby('hqlatype')[['Pre-Haircut', 'Post-Haircut']].sum().reset_index()
    grouped.columns = ['HQLA Type', 'Pre-Haircut', 'Post-Haircut']

    return grouped
    
hqla_df = get_hqla_treemap_data(scenario_id=scenario_id)

# Pre-haircut
fig_pre = px.treemap(
    hqla_df,
    path=['HQLA Type'],
    values='Pre-Haircut',
    title="HQLA Composition (Pre-Haircut)"
)

fig_pre.update_traces(
    textinfo="label+percent entry",   # Show label and percentage
    hovertemplate=''                  # Suppress hover box
)

# Same for post-haircut chart
fig_post = px.treemap(
    hqla_df,
    path=['HQLA Type'],
    values='Post-Haircut',
    title="HQLA Composition (Post-Haircut)"
)

fig_post.update_traces(
    textinfo="label+percent entry",
    hovertemplate=''
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_pre, use_container_width=True)

with col2:
    st.plotly_chart(fig_post, use_container_width=True)


# ==========================================================
# NSFR Structure (Bars)
# ==========================================================
st.subheader("NSFR Funding Structure")

# Bar chart: ASF vs RSF breakdown
asf_components = nsfr.get("ASF_components", {})
rsf_components = nsfr.get("RSF_components", {})

asf_labels = list(asf_components.keys())
asf_values = list(asf_components.values())

rsf_labels = list(rsf_components.keys())
rsf_values = list(rsf_components.values())

# Make lengths match for plotting
max_len = max(len(asf_labels), len(rsf_labels))
asf_labels += [''] * (max_len - len(asf_labels))
asf_values += [0] * (max_len - len(asf_values))
rsf_labels += [''] * (max_len - len(rsf_labels))
rsf_values += [0] * (max_len - len(rsf_values))

# Use the same scenario_id as in your dashboard
cashflows = queries.get_cashflows(scenario_id=scenario_id)

# ASF Weights
asf_weights_df = (
    cashflows[cashflows['direction'] == 'inflow']
    .groupby('product')['asf_factor']
    .mean()
    .apply(lambda x: f"{int(x * 100)}%")
    .to_dict()
)

# RSF Weights
rsf_weights_df = (
    cashflows[cashflows['direction'] == 'outflow']
    .groupby('product')['rsf_factor']
    .mean()
    .apply(lambda x: f"{int(x * 100)}%")
    .to_dict()
)


fig = go.Figure()

fig.add_trace(go.Bar(
    x=asf_labels,
    y=asf_values,
    name='ASF (Available Stable Funding)',
    marker_color='green',
    hovertext=[f"EBA Weight: {asf_weights_df.get(label, '')}" for label in asf_labels],
    hoverinfo='text+y'  # show both custom hovertext and y-value
))

fig.add_trace(go.Bar(
    x=rsf_labels,
    y=rsf_values,
    name='RSF (Required Stable Funding)',
    marker_color='red',
    hovertext=[f"EBA Weight: {rsf_weights_df.get(label, '')}" for label in rsf_labels],
    hoverinfo='text+y'
))

fig.update_layout(
    barmode='group',
    title='ASF vs RSF Breakdown with EBA Weightings',
    xaxis_title='Funding / Asset Categories',
    yaxis_title='EUR',
    xaxis_tickangle=-30,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Cashflow Heatmap
# ==========================================================
st.subheader("Cashflow Gap Heatmap")
st.caption("Net cashflows across maturity buckets. Inflows capped to 75% of outflows per EBA LCR rules.")

# Load data
pivot_df = compute.calculate_cashflow_gap_heatmap(scenario_id=scenario_id)

import plotly.graph_objects as go

# Ensure all axis labels are strings
x_labels = pivot_df.columns.astype(str)
y_labels = pivot_df.index.astype(str)

heatmap = go.Figure(data=go.Heatmap(
    z=-pivot_df.values / 1e3,
    x=pivot_df.columns,
    y=pivot_df.index,
    colorscale='RdYlGn_r',
    zmin=-1200,
    zmax=1200,
    hovertemplate="Date: %{x}<br>Bucket: %{y}<br>Net Flow: %{z:,.0f} kEUR"
))

# Full expected order
expected_order = ['0-7d', '8-30d', '31-90d', '91-180d', '181-365d', '>1y']
# Actual buckets present in the data
present_buckets = [b for b in expected_order if b in pivot_df.index.tolist()]
# Then apply to update_layout
heatmap.update_layout(
    yaxis=dict(
        categoryorder="array",
        categoryarray=present_buckets
    )
)

st.plotly_chart(heatmap, use_container_width=True)

# ==========================================================
# Dual Axis Plot (LCR vs Net Cashflow)
# ==========================================================

st.subheader("Dual Axis Plot (LCR vs Net Cashflow)")


# 👇 Scenario Selector
scenario_label = st.selectbox(
    "Select Scenario",
    ["Baseline", "Stress: ECB Stress", "Stress: Liquidity Shock"]
)

scenario_map = {
    "Baseline": None,
    "Stress: ECB Stress": 1,
    "Stress: Liquidity Shock": 2
}

scenario_id = scenario_map[scenario_label]


lcr_df = compute.calculate_lcr_timeseries(scenario_id=scenario_id)

dual_axis_fig = go.Figure()

dual_axis_fig.add_trace(go.Bar(
    x=lcr_df['date'],
    y=lcr_df['net_cashflow'],
    name="Net Cashflow",
    marker_color="orange",
    yaxis="y1"
))

dual_axis_fig.add_trace(go.Scatter(
    x=lcr_df['date'],
    y=lcr_df['lcr'],
    name="LCR",
    mode='lines+markers',
    line=dict(color='blue'),
    yaxis="y2"
))

dual_axis_fig.add_trace(go.Scatter(
    x=[lcr_df['date'].min(), lcr_df['date'].max()],
    y=[100, 100],
    mode='lines',
    name='LCR Threshold (100%)',
    line=dict(color='red', dash='dot'),
    yaxis='y2',
    showlegend=True
))

dual_axis_fig.update_layout(
    title="Daily Net Cashflows vs. LCR Ratio",
    xaxis=dict(title="Date"),
    yaxis=dict(
        title="Net Cashflow",
        side='left',
        showgrid=False,
        rangemode="tozero"  # Optional: ensure baseline is included
    ),
    yaxis2=dict(
        title="LCR Ratio",
        overlaying='y',
        side='right',
        showgrid=False,
        range=[0, max(1.5, lcr_df['lcr'].max() * 1.1)]  # Cap at 1.5 or 10% above max
    ),
    legend=dict(x=0.01, y=1),
    height=400
)

st.plotly_chart(dual_axis_fig, use_container_width=True)

# ==========================================================
# LCR/NSFR Over Time Line Chart
# ==========================================================

st.subheader("LCR & NSFR Over Time")

# --- Scenario Selector
scenario_label = st.selectbox(
    "Select Scenario",
    ["Baseline", "Stress: ECB Stress", "Stress: Liquidity Shock"],
    key="liquidity_scenario"
)

scenario_map = {
    "Baseline": None,
    "Stress: ECB Stress": 1,
    "Stress: Liquidity Shock": 2
}
scenario_id = scenario_map[scenario_label]

# --- Load data
lcr_df = compute.calculate_lcr_timeseries(scenario_id=scenario_id)
nsfr_df = compute.calculate_nsfr_timeseries(scenario_id=scenario_id)

# --- Merge data on date
combined = pd.merge(
    lcr_df[['date', 'lcr']],
    nsfr_df[['date', 'NSFR']],
    on='date',
    how='outer'
).sort_values('date')

# --- Create dual-axis plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=combined['date'],
    y=combined['lcr'],
    name='LCR',
    mode='lines+markers',
    line=dict(color='blue'),
    yaxis='y1'
))

fig.add_trace(go.Scatter(
    x=combined['date'],
    y=combined['NSFR'],
    name='NSFR',
    mode='lines+markers',
    line=dict(color='green'),
    yaxis='y2'
))

# Threshold lines
fig.add_shape(
    type='line',
    x0=combined['date'].min(), x1=combined['date'].max(),
    y0=1.0, y1=1.0,
    line=dict(color='red', dash='dash'),
    yref='y2'
)

fig.add_shape(
    type='line',
    x0=combined['date'].min(), x1=combined['date'].max(),
    y0=100.0, y1=100.0,
    line=dict(color='red', dash='dash'),
    yref='y1'
)

# Layout
fig.update_layout(
    title=f"Liquidity Ratios Over Time — {scenario_label}",
    xaxis=dict(title="Date"),
    yaxis=dict(title="LCR (%)", side='left', showgrid=False),
    yaxis2=dict(title="NSFR (Ratio)", overlaying='y', side='right', showgrid=False),
    legend=dict(x=0.01, y=1),
    height=450
)

st.plotly_chart(fig, use_container_width=True)

