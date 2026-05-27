import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src import compute, queries
from src.lineage import render_model_lineage
from src.scenario import scenario_sidebar

st.set_page_config(page_title="Liquidity Risk", layout="wide")
st.title("Liquidity Risk")

scenario_id = scenario_sidebar()
scenario_name = st.session_state.get("scenario_choice", "Baseline")

render_model_lineage(expanded=False)

# KPIs
lcr = compute.calculate_lcr(scenario_id)
nsfr = compute.calculate_nsfr(scenario_id)

st.subheader(f"Scenario: {scenario_name}")
k1, k2 = st.columns(2)
k1.metric("LCR", f"{lcr['LCR']:.2f}")
k2.metric("NSFR", f"{nsfr['NSFR']:.2f}")

# ==========================================================
# ALMM survival horizon (Phase 2.1d)
# ==========================================================
st.subheader("ALMM survival horizon")
st.caption(
    "Days the bank can survive cumulative net outflows before its counterbalancing "
    "capacity (HQLA stock + subsequent net receipts) is exhausted. Three preset "
    "stresses follow ALMM template C66: idiosyncratic (bank-specific shock), "
    "market-wide (wholesale funding freeze, HQLA haircuts widen), and combined."
)

surv_df = queries.get_mart("mart_survival_horizon", scenario_id=scenario_id)
ladder_df = queries.get_mart("mart_cbc_ladder", scenario_id=scenario_id)

if surv_df.empty:
    st.info("Run `python -m basel_risk_engine.run` to populate the ALMM survival marts.")
else:
    stress_order = ["idiosyncratic", "market_wide", "combined"]
    surv_df = surv_df.set_index("stress_name").reindex(stress_order).reset_index()
    cols = st.columns(len(stress_order))
    pretty = {
        "idiosyncratic": "Idiosyncratic",
        "market_wide": "Market-wide",
        "combined": "Combined",
    }
    for col, (_, row) in zip(cols, surv_df.iterrows()):
        with col:
            horizon = int(row["survival_horizon_days"])
            label = pretty.get(row["stress_name"], row["stress_name"])
            st.metric(
                label,
                f"{horizon} d" + (" (no breach)" if not row["is_breached"] else ""),
                delta=row["severity_bucket"],
                delta_color="inverse" if row["is_breached"] else "normal",
            )
            st.caption(
                f"Initial CBC: {row['initial_cbc']:,.0f} EUR · "
                f"Peak deficit: {row['peak_deficit']:,.0f} EUR"
            )

    # ---- Counterbalancing-capacity trajectory (the survival curve) -------
    if not ladder_df.empty:
        color_map = {
            "idiosyncratic": "#1f77b4",
            "market_wide": "#ff7f0e",
            "combined": "#d62728",
        }
        fig = go.Figure()
        for s in stress_order:
            sub = ladder_df[ladder_df["stress_name"] == s].sort_values("day_offset")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["day_offset"],
                y=sub["running_cbc"],
                mode="lines",
                name=pretty[s],
                line=dict(color=color_map[s], width=2),
                hovertemplate="day %{x}<br>running CBC: %{y:,.0f} EUR<extra></extra>",
            ))
            # Survival horizon marker (only if breached)
            if surv_df[surv_df["stress_name"] == s]["is_breached"].any():
                h = int(surv_df[surv_df["stress_name"] == s]["survival_horizon_days"].iloc[0])
                fig.add_trace(go.Scatter(
                    x=[h], y=[sub.loc[sub["day_offset"] == h, "running_cbc"].iloc[0]],
                    mode="markers",
                    name=f"{pretty[s]} breach",
                    marker=dict(symbol="x", size=12, color=color_map[s], line=dict(width=2)),
                    showlegend=False,
                    hovertemplate=f"survival horizon: day {h}<extra></extra>",
                ))
        # Zero line — the survival threshold
        fig.add_hline(y=0, line_dash="dash", line_color="grey",
                      annotation_text="CBC exhausted", annotation_position="top right")
        fig.update_layout(
            title="Counterbalancing capacity trajectory",
            xaxis_title="Days from valuation date",
            yaxis_title="Running CBC (EUR)",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
            hovermode="x unified",
        )
        st.plotly_chart(fig, width='stretch')
        st.caption(
            "Each curve starts at the stressed HQLA stock and is driven down by "
            "stressed net outflows (deposit runoff, wholesale rollover failure, "
            "asset-inflow credit haircuts, LCR-style 75% inflow cap). The first "
            "crossing of zero is the survival horizon — the 'x' marker."
        )

# ==========================================================
# LCR Waterfall
# ==========================================================
st.subheader("LCR Waterfall")

h = lcr["HQLA"]
out = -lcr["Outflows"]
cap_in = min(lcr["Inflows"], lcr["Outflows"] * 0.75)
net_out = lcr["NetOutflows"]

fig = go.Figure(
    go.Waterfall(
        name="LCR",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["HQLA", "Outflows", "Inflows (capped)", "Net Outflows"],
        y=[h, out, cap_in, net_out],
        textposition="outside",
        text=[f"{h:,.0f}", f"{out:,.0f}", f"{cap_in:,.0f}", f"{net_out:,.0f}"],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    )
)
fig.update_layout(title="LCR Waterfall Breakdown", yaxis_title="EUR", waterfallgap=0.3)
st.plotly_chart(fig, width='stretch')

# ==========================================================
# HQLA Composition (Pre/Post Haircut)
# ==========================================================
def hqla_treemap_data(scenario_id):
    cashflows = queries.get_cashflows(scenario_id=scenario_id)
    params = queries.get_params()
    haircut_map = {
        "Level1": 0.0,
        "Level2A": float(params.get("haircut_level2a", 0.15)),
        "Level2B": float(params.get("haircut_level2b", 0.5)),
        "None": 1.0,
    }
    hqla = cashflows[cashflows["hqlatype"].isin(["Level1", "Level2A", "Level2B"])].copy()
    hqla["Pre-Haircut"] = hqla["amount"]
    hqla["Post-Haircut"] = hqla["amount"] * hqla["hqlatype"].map(
        lambda h: 1 - haircut_map.get(h, 1)
    )
    grouped = hqla.groupby("hqlatype")[["Pre-Haircut", "Post-Haircut"]].sum().reset_index()
    grouped.columns = ["HQLA Type", "Pre-Haircut", "Post-Haircut"]
    return grouped


hqla_df = hqla_treemap_data(scenario_id)

fig_pre = px.treemap(
    hqla_df, path=["HQLA Type"], values="Pre-Haircut", title="HQLA Composition (Pre-Haircut)"
)
fig_pre.update_traces(textinfo="label+percent entry", hovertemplate="")
fig_post = px.treemap(
    hqla_df, path=["HQLA Type"], values="Post-Haircut", title="HQLA Composition (Post-Haircut)"
)
fig_post.update_traces(textinfo="label+percent entry", hovertemplate="")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_pre, width='stretch')
with col2:
    st.plotly_chart(fig_post, width='stretch')

# ==========================================================
# NSFR Funding Structure
# ==========================================================
st.subheader("NSFR Funding Structure")

asf_components = nsfr.get("ASF_components", {})
rsf_components = nsfr.get("RSF_components", {})

asf_labels = list(asf_components.keys())
asf_values = list(asf_components.values())
rsf_labels = list(rsf_components.keys())
rsf_values = list(rsf_components.values())

max_len = max(len(asf_labels), len(rsf_labels))
asf_labels += [""] * (max_len - len(asf_labels))
asf_values += [0] * (max_len - len(asf_values))
rsf_labels += [""] * (max_len - len(rsf_labels))
rsf_values += [0] * (max_len - len(rsf_values))

# ASF / RSF weights by product (filtered by which factor is non-zero,
# matching compute.calculate_nsfr's interpretation)
cashflows = queries.get_cashflows(scenario_id=scenario_id)
asf_weights = (
    cashflows[cashflows["asf_factor"] > 0]
    .groupby("product")["asf_factor"]
    .mean()
    .apply(lambda x: f"{int(x * 100)}%")
    .to_dict()
)
rsf_weights = (
    cashflows[cashflows["rsf_factor"] > 0]
    .groupby("product")["rsf_factor"]
    .mean()
    .apply(lambda x: f"{int(x * 100)}%")
    .to_dict()
)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=asf_labels,
        y=asf_values,
        name="ASF (Available Stable Funding)",
        marker_color="green",
        hovertext=[f"EBA Weight: {asf_weights.get(label, '')}" for label in asf_labels],
        hoverinfo="text+y",
    )
)
fig.add_trace(
    go.Bar(
        x=rsf_labels,
        y=rsf_values,
        name="RSF (Required Stable Funding)",
        marker_color="red",
        hovertext=[f"EBA Weight: {rsf_weights.get(label, '')}" for label in rsf_labels],
        hoverinfo="text+y",
    )
)
fig.update_layout(
    barmode="group",
    title="ASF vs RSF Breakdown with EBA Weightings",
    xaxis_title="Funding / Asset Categories",
    yaxis_title="EUR",
    xaxis_tickangle=-30,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig, width='stretch')

# ==========================================================
# Cashflow Gap Heatmap
# ==========================================================
st.subheader("Cashflow Gap Heatmap")
st.caption(
    "Net cashflows across maturity buckets. Inflows capped to 75% of outflows per EBA LCR rules."
)

pivot_df = compute.calculate_cashflow_gap_heatmap(scenario_id=scenario_id)

heatmap = go.Figure(
    data=go.Heatmap(
        z=-pivot_df.values / 1e3,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn_r",
        zmin=-1200,
        zmax=1200,
        hovertemplate="Date: %{x}<br>Bucket: %{y}<br>Net Flow: %{z:,.0f} kEUR",
    )
)

expected_order = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">1y"]
present_buckets = [b for b in expected_order if b in pivot_df.index.tolist()]
heatmap.update_layout(
    yaxis=dict(categoryorder="array", categoryarray=present_buckets)
)
st.plotly_chart(heatmap, width='stretch')

# ==========================================================
# LCR vs Net Cashflow (dual-axis)
# ==========================================================
st.subheader("LCR vs Net Cashflow")

lcr_df = compute.calculate_lcr_timeseries(scenario_id=scenario_id)

dual_axis_fig = go.Figure()
dual_axis_fig.add_trace(
    go.Bar(
        x=lcr_df["date"],
        y=lcr_df["net_cashflow"],
        name="Net Cashflow",
        marker_color="orange",
        yaxis="y1",
    )
)
dual_axis_fig.add_trace(
    go.Scatter(
        x=lcr_df["date"],
        y=lcr_df["lcr"],
        name="LCR",
        mode="lines+markers",
        line=dict(color="blue"),
        yaxis="y2",
    )
)
dual_axis_fig.add_trace(
    go.Scatter(
        x=[lcr_df["date"].min(), lcr_df["date"].max()],
        y=[1.0, 1.0],
        mode="lines",
        name="LCR Threshold (100%)",
        line=dict(color="red", dash="dot"),
        yaxis="y2",
        showlegend=True,
    )
)
dual_axis_fig.update_layout(
    title="Daily Net Cashflows vs. LCR Ratio",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Net Cashflow", side="left", showgrid=False, rangemode="tozero"),
    yaxis2=dict(
        title="LCR Ratio",
        overlaying="y",
        side="right",
        showgrid=False,
        range=[0, max(1.5, lcr_df["lcr"].max() * 1.1)],
    ),
    legend=dict(x=0.01, y=1),
    height=400,
)
st.plotly_chart(dual_axis_fig, width='stretch')

# ==========================================================
# LCR & NSFR Over Time
# ==========================================================
st.subheader("LCR & NSFR Over Time")

nsfr_df = compute.calculate_nsfr_timeseries(scenario_id=scenario_id)

combined = pd.merge(
    lcr_df[["date", "lcr"]], nsfr_df[["date", "NSFR"]], on="date", how="outer"
).sort_values("date")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=combined["date"],
        y=combined["lcr"],
        name="LCR",
        mode="lines+markers",
        line=dict(color="blue"),
        yaxis="y1",
    )
)
fig.add_trace(
    go.Scatter(
        x=combined["date"],
        y=combined["NSFR"],
        name="NSFR",
        mode="lines+markers",
        line=dict(color="green"),
        yaxis="y2",
    )
)

# Threshold lines
fig.add_shape(
    type="line",
    x0=combined["date"].min(),
    x1=combined["date"].max(),
    y0=1.0,
    y1=1.0,
    line=dict(color="red", dash="dash"),
    yref="y2",
)
fig.add_shape(
    type="line",
    x0=combined["date"].min(),
    x1=combined["date"].max(),
    y0=1.0,
    y1=1.0,
    line=dict(color="red", dash="dash"),
    yref="y1",
)

fig.update_layout(
    title=f"Liquidity Ratios Over Time — {scenario_name}",
    xaxis=dict(title="Date"),
    yaxis=dict(title="LCR", side="left", showgrid=False),
    yaxis2=dict(title="NSFR", overlaying="y", side="right", showgrid=False),
    legend=dict(x=0.01, y=1),
    height=450,
)
st.plotly_chart(fig, width='stretch')
