import _bootstrap  # noqa: F401  -- must precede `src`/`basel_common` imports

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src import compute
from src.lineage import render_model_lineage
from src.scenario import scenario_sidebar

st.set_page_config(page_title="RWA and Capital Adequacy", layout="wide")
st.title("RWA and Capital Adequacy")

scenario_id = scenario_sidebar()

render_model_lineage(expanded=False)

# ==========================================================
# RWA Breakdown Treemap
# ==========================================================
st.subheader("RWA Breakdown by Asset Class")

rwa_df = compute.calculate_rwa_by_approach_and_asset_class(scenario_id=scenario_id)

std_rwa = rwa_df.loc[rwa_df["approach"] == "STD", "rwa_amount"].sum()
irb_rwa = rwa_df.loc[rwa_df["approach"] == "IRB", "rwa_amount"].sum()
output_floor = 0.725 * std_rwa

if irb_rwa < output_floor:
    st.error(
        f"⚠️ IRB Output Floor Binding: IRB RWA ({irb_rwa:,.0f}) < 72.5% of STD RWA ({output_floor:,.0f})"
    )
else:
    st.success(
        f"✅ IRB RWA ({irb_rwa:,.0f}) complies with 72.5% output floor ({output_floor:,.0f})"
    )

fig = px.treemap(
    rwa_df,
    path=["approach", "asset_class"],
    values="rwa_amount",
    title="RWA by Approach and Asset Class",
    color="rwa_amount",
    color_continuous_scale="Blues",
)
fig.update_traces(textinfo="label+value+percent entry")
st.plotly_chart(fig, width='stretch')

# ==========================================================
# Capital Ratios Over Time
# ==========================================================
st.subheader("Capital Ratios Over Time")

capital_ts = compute.calculate_capital_timeseries(scenario_id=scenario_id)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=capital_ts["date"],
        y=capital_ts["CET1 Ratio"],
        name="CET1 Ratio",
        mode="lines+markers",
        line=dict(color="#1f77b4"),
    )
)
fig.add_trace(
    go.Scatter(
        x=capital_ts["date"],
        y=capital_ts["Tier1 Ratio"],
        name="Tier1 Ratio",
        mode="lines+markers",
        line=dict(color="#ff7f0e"),
    )
)
fig.add_trace(
    go.Scatter(
        x=capital_ts["date"],
        y=capital_ts["Total Capital Ratio"],
        name="Total Capital Ratio",
        mode="lines+markers",
        line=dict(color="#2ca02c"),
    )
)

# Pillar 1 minima (decimal ratios): CET1 4.5%, Tier1 6%, Total 8%.
# CET1 + capital conservation buffer = 7%. Annotations parked bottom-left
# so they don't overlap the ratio lines (which sit around 12-17%).
fig.add_hline(y=0.045, line_dash="dot", line_color="grey",
              annotation_text="CET1 min (4.5%)", annotation_position="bottom left")
fig.add_hline(y=0.06, line_dash="dot", line_color="grey",
              annotation_text="Tier1 min (6%)", annotation_position="bottom left")
fig.add_hline(y=0.07, line_dash="dash", line_color="grey",
              annotation_text="CET1 + buffer (7%)", annotation_position="bottom left")
fig.add_hline(y=0.08, line_dash="dot", line_color="grey",
              annotation_text="Total min (8%)", annotation_position="bottom left")

fig.update_layout(
    title="Capital Ratios vs. Regulatory Thresholds",
    xaxis_title="Date",
    yaxis_title="Capital Ratio",
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    hovermode="x unified",
)
fig.update_yaxes(tickformat=".1%", rangemode="tozero")
fig.update_traces(hovertemplate="%{x|%b %d, %Y}<br>%{y:.2%}")
st.plotly_chart(fig, width='stretch')

# ==========================================================
# RWA Sensitivity Slider
# ==========================================================
st.subheader("Capital Ratios Under RWA Stress")

shock_pct = (
    st.slider(
        "Simulate RWA Increase (%)",
        min_value=0,
        max_value=100,
        step=5,
        value=0,
        key="rwa_stress_slider",
    )
    / 100
)

ratios_shocked = compute.calculate_capital_ratios_under_rwa_shock(
    rwa_shock_pct=shock_pct, scenario_id=scenario_id
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Shocked RWA", f"{ratios_shocked['RWA (shocked)'] / 1e9:,.2f} B EUR")
col2.metric("CET1 Ratio", f"{ratios_shocked['CET1 Ratio']:.2%}")
col3.metric("Tier1 Ratio", f"{ratios_shocked['Tier1 Ratio']:.2%}")
col4.metric("Total Capital Ratio", f"{ratios_shocked['Total Capital Ratio']:.2%}")
