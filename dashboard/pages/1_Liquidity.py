import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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


st.subheader("HQLA composition — pre vs post regulatory haircut")
hqla_df = hqla_treemap_data(scenario_id)
haircut_pct = {"Level1": 0, "Level2A": 15, "Level2B": 50}
hqla_df["y_label"] = hqla_df["HQLA Type"].map(
    lambda t: f"{t} (haircut {haircut_pct.get(t, 0)}%)"
)
hqla_melt = hqla_df.melt(
    id_vars=["y_label"],
    value_vars=["Pre-Haircut", "Post-Haircut"],
    var_name="Stage",
    value_name="Amount",
)
fig_hqla = px.bar(
    hqla_melt,
    x="Amount",
    y="y_label",
    color="Stage",
    orientation="h",
    barmode="group",
    color_discrete_map={"Pre-Haircut": "#9ecae1", "Post-Haircut": "#1f77b4"},
    text="Amount",
    labels={"Amount": "HQLA stock (EUR)", "y_label": ""},
)
fig_hqla.update_traces(texttemplate="%{x:,.0f}", textposition="outside")
fig_hqla.update_layout(
    height=340,
    yaxis=dict(
        categoryorder="array",
        categoryarray=[
            f"Level2B (haircut 50%)",
            f"Level2A (haircut 15%)",
            f"Level1 (haircut 0%)",
        ],
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    margin=dict(l=160, r=120),
)
st.plotly_chart(fig_hqla, width='stretch')
st.caption(
    "Level1 is uncut; Level2A is shaved 15%; Level2B is halved. Post-haircut HQLA is "
    "the denominator-relevant figure for LCR."
)

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
# Cashflow Gap Heatmap (+ daily net totals strip)
# ==========================================================
st.subheader("Cashflow Gap Heatmap")
st.caption(
    "Net cashflows by maturity bucket × date — green = inflow surplus, "
    "red = outflow surplus. Top strip is the daily net across all buckets."
)

pivot_df = compute.calculate_cashflow_gap_heatmap(scenario_id=scenario_id)
expected_order = ["0-7d", "8-30d", "31-90d", "91-180d", "181-365d", ">1y"]
present_buckets = [b for b in expected_order if b in pivot_df.index.tolist()]
pivot_df = pivot_df.reindex(present_buckets)

z_keuro = pivot_df.values / 1e3                                  # signed kEUR: +inflow / -outflow
daily_totals = pivot_df.sum(axis=0) / 1e3                        # column-wise net (kEUR)
zlim = max(abs(z_keuro.min()), abs(z_keuro.max()), 1.0)          # symmetric, data-driven

cf_fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04,
    row_heights=[0.18, 0.82],
)
# Top strip: daily net totals (single-row heatmap)
cf_fig.add_trace(
    go.Heatmap(
        z=[daily_totals.values],
        x=pivot_df.columns,
        y=["Daily net"],
        colorscale="RdYlGn",
        zmid=0,
        zmin=-abs(daily_totals).max() if abs(daily_totals).max() > 0 else -1,
        zmax= abs(daily_totals).max() if abs(daily_totals).max() > 0 else  1,
        showscale=False,
        hovertemplate="%{x|%b %d, %Y}<br>Net (all buckets): %{z:,.0f} kEUR<extra></extra>",
    ),
    row=1, col=1,
)
# Main heatmap: bucket × date
cf_fig.add_trace(
    go.Heatmap(
        z=z_keuro,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale="RdYlGn",
        zmid=0,
        zmin=-zlim,
        zmax= zlim,
        colorbar=dict(title="Net flow<br>(kEUR)", tickformat=",.0f"),
        hovertemplate="%{x|%b %d, %Y}<br>Bucket: %{y}<br>Net: %{z:,.0f} kEUR<extra></extra>",
    ),
    row=2, col=1,
)
cf_fig.update_yaxes(
    categoryorder="array",
    categoryarray=present_buckets[::-1],   # shortest bucket on top
    row=2, col=1,
)
cf_fig.update_xaxes(tickformat="%b %d", row=2, col=1)
cf_fig.update_layout(height=440, margin=dict(t=20, b=40))
st.plotly_chart(cf_fig, width='stretch')

# ==========================================================
# LCR + Net Cashflow (stacked subplots — shared x-axis)
# ==========================================================
st.subheader("LCR + Net Cashflow")

lcr_df = compute.calculate_lcr_timeseries(scenario_id=scenario_id)
# Warm-up gate nulls the LCR for the first 29 days of each scenario; drop
# those rows so the x-axis starts at the first valid ratio rather than
# leading with 29 days of bars under an empty line.
lcr_df = lcr_df[lcr_df["lcr"].notna()].reset_index(drop=True)

cf_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in lcr_df["net_cashflow"]]

ts_fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.6, 0.4],
    subplot_titles=(
        "LCR — 30-day backward rolling",
        "Daily net cashflow (inflows − outflows)",
    ),
)
ts_fig.add_trace(
    go.Scatter(
        x=lcr_df["date"], y=lcr_df["lcr"],
        mode="lines+markers", name="LCR",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>LCR: %{y:.2f}<extra></extra>",
    ),
    row=1, col=1,
)
ts_fig.add_hline(
    y=1.0, line_dash="dot", line_color="red",
    annotation_text="100% threshold", annotation_position="top right",
    row=1, col=1,
)
ts_fig.add_trace(
    go.Bar(
        x=lcr_df["date"], y=lcr_df["net_cashflow"],
        marker_color=cf_colors, name="Net cashflow",
        hovertemplate="%{x|%b %d, %Y}<br>Net: %{y:,.0f} EUR<extra></extra>",
    ),
    row=2, col=1,
)
ts_fig.add_hline(y=0, line_color="grey", line_width=1, row=2, col=1)
ts_fig.update_yaxes(title_text="LCR ratio", row=1, col=1)
ts_fig.update_yaxes(title_text="EUR", row=2, col=1)
ts_fig.update_xaxes(
    title_text="Date", row=2, col=1,
    tickformat="%b %d", tickangle=-30, nticks=10,
)
ts_fig.update_layout(height=500, showlegend=False, margin=dict(t=60, b=60))
st.plotly_chart(ts_fig, width='stretch')

# ==========================================================
# LCR & NSFR Over Time (single y-axis — both are dimensionless ratios
# with the same 100% threshold)
# ==========================================================
st.subheader("LCR & NSFR Over Time")

nsfr_df = compute.calculate_nsfr_timeseries(scenario_id=scenario_id)
nsfr_df = nsfr_df[nsfr_df["NSFR"].notna()].reset_index(drop=True)

# lcr_df was already filtered to non-null rows above. Inner-join on date so
# the chart's x-axis starts at the first date where both ratios are valid.
combined = pd.merge(
    lcr_df[["date", "lcr"]], nsfr_df[["date", "NSFR"]], on="date", how="inner"
).sort_values("date")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=combined["date"], y=combined["lcr"],
        name="LCR", mode="lines+markers",
        line=dict(color="#1f77b4"),
        hovertemplate="%{x|%b %d, %Y}<br>LCR: %{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=combined["date"], y=combined["NSFR"],
        name="NSFR", mode="lines+markers",
        line=dict(color="#2ca02c"),
        hovertemplate="%{x|%b %d, %Y}<br>NSFR: %{y:.2f}<extra></extra>",
    )
)
fig.add_hline(
    y=1.0, line_dash="dot", line_color="red",
    annotation_text="100% threshold", annotation_position="top right",
)
fig.update_layout(
    title=f"Liquidity ratios over time — {scenario_name}",
    xaxis=dict(
        title="Date",
        tickformat="%b %d",
        tickangle=-30,
        nticks=10,
    ),
    yaxis_title="Ratio",
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
    hovermode="x unified",
    margin=dict(b=60),
)
st.plotly_chart(fig, width='stretch')
