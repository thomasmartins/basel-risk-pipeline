import streamlit as st
import plotly.express as px
from src import compute, queries
import plotly.graph_objects as go

st.set_page_config(layout="wide")


st.title("RWA and Capital Adequacy")

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
# RWA Breakdown Treemap
# ==========================================================

st.subheader("RWA Breakdown by Asset Class")

rwa_df = compute.calculate_rwa_by_approach_and_asset_class(scenario_id=scenario_id)

std_rwa = rwa_df[rwa_df['approach'] == 'STD']['rwa_amount'].sum()
irb_rwa = rwa_df[rwa_df['approach'] == 'IRB']['rwa_amount'].sum()
output_floor = 0.725 * std_rwa

# Display diagnostic message
if irb_rwa < output_floor:
    st.error(f"⚠️ IRB Output Floor Binding: IRB RWA ({irb_rwa:,.0f}) < 72.5% of STD RWA ({output_floor:,.0f})")
else:
    st.success(f"✅ IRB RWA ({irb_rwa:,.0f}) complies with 72.5% output floor ({output_floor:,.0f})")


fig = px.treemap(
    rwa_df,
    path=['approach', 'asset_class'],
    values='rwa_amount',
    title="RWA by Approach and Asset Class",
    color='rwa_amount',
    color_continuous_scale='Blues',
)

fig.update_traces(textinfo='label+value+percent entry')

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# CET1 and Tier 1 Time Series
# ==========================================================

st.subheader("Capital Ratios Over Time")

capital_ts = compute.calculate_capital_timeseries()

fig = go.Figure()

# CET1 line
fig.add_trace(go.Scatter(
    x=capital_ts['date'],
    y=capital_ts['CET1 Ratio'],
    name="CET1 Ratio",
    mode='lines+markers',
    line=dict(color='blue')
))

# Tier1 line
fig.add_trace(go.Scatter(
    x=capital_ts['date'],
    y=capital_ts['Tier1 Ratio'],
    name="Tier1 Ratio",
    mode='lines+markers',
    line=dict(color='orange')
))

# --- Thresholds ---
fig.add_hline(y=4.5, line_dash="dot", line_color="white", annotation_text="CET1 Min (4.5%)")
fig.add_hline(y=6.0, line_dash="dot", line_color="white", annotation_text="Tier1 Min (6%)")
fig.add_hline(y=7.0, line_dash="dash", line_color="blue", annotation_text="CET1 + Buffer")

# --- Layout ---
fig.update_layout(
    title="CET1 and Tier1 Ratios vs. Regulatory Thresholds",
    xaxis_title="Date",
    yaxis_title="Capital Ratio (%)",
    height=450,
    legend=dict(x=0.01, y=1)
)

# Keep y-axis as raw numbers (not %) since your data is in percent already
fig.update_yaxes(tickformat=".0f")
fig.update_traces(hovertemplate='%{x|%b %d, %Y}<br>%{y:.2f}%')

st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# RWA Sensitivity Slider
# ==========================================================

st.subheader("Capital Ratios Under RWA Stress")

# Slider: RWA Shock (e.g., downgrade)
shock_pct = st.slider(
    "Simulate RWA Increase (%)",
    min_value=0,
    max_value=100,
    step=5,
    value=0,
    key="rwa_stress_slider"
) / 100

ratios_shocked = compute.calculate_capital_ratios_under_rwa_shock(rwa_shock_pct=shock_pct, scenario_id=scenario_id)

# Show metrics
col1, col2, col3, col4 = st.columns(4)
rwa_billion = ratios_shocked['RWA (shocked)'] / 1e9
col1.metric("Shocked RWA", f"{rwa_billion:,.2f} B EUR")
col2.metric("CET1 Ratio", f"{ratios_shocked['CET1 Ratio']:.2%}")
col3.metric("Tier1 Ratio", f"{ratios_shocked['Tier1 Ratio']:.2%}")
col4.metric("Total Capital Ratio", f"{ratios_shocked['Total Capital Ratio']:.2%}")