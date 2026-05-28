import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src import compute
from src.lineage import render_model_lineage
from src.scenario import scenario_sidebar

st.set_page_config(page_title="Stress Testing Panel", layout="wide")
st.title("Stress Testing Panel")

scenario_id = scenario_sidebar()
scenario_name = st.session_state.get("scenario_choice", "Baseline")

render_model_lineage(expanded=False)

# --- Sidebar Inputs ---
st.sidebar.header("Stress Test Parameters")
shock_bps = st.sidebar.slider("Interest Rate Shock (bps)", -300, 300, 200, step=25)
retail_withdrawal_pct = st.sidebar.slider("Retail Withdrawal (%)", 0.0, 1.0, 0.2, step=0.05)
wholesale_withdrawal_pct = st.sidebar.slider("Wholesale Withdrawal (%)", 0.0, 1.0, 0.4, step=0.05)
rwa_stress_pct = st.sidebar.slider("RWA Increase (%)", 0.0, 1.0, 0.1, step=0.05)

results = compute.run_stress_test(
    shock_bps=shock_bps,
    retail_withdrawal_pct=retail_withdrawal_pct,
    wholesale_withdrawal_pct=wholesale_withdrawal_pct,
    rwa_stress_pct=rwa_stress_pct,
    scenario_id=scenario_id,
)

st.subheader(f"Key Risk Metrics — Scenario: {scenario_name}")

col1, col2, col3 = st.columns(3)
col1.metric(
    "LCR",
    f"{results['LCR (Stressed)']:.2f}",
    f"{results['LCR (Stressed)'] - results['LCR (Base)']:+.2f}",
)
col2.metric(
    "NSFR",
    f"{results['NSFR (Stressed)']:.2f}",
    f"{results['NSFR (Stressed)'] - results['NSFR (Base)']:+.2f}",
)
col3.metric("∆EVE", f"{results['∆EVE (Stressed)']:,.2f} EUR")

col4, col5, col6 = st.columns(3)
col4.metric(
    "CET1 Ratio",
    f"{results['CET1 Ratio (Stressed)']:.2%}",
    f"{results['CET1 Ratio (Stressed)'] - results['CET1 Ratio (Base)']:+.2%}",
)
col5.metric(
    "Tier1 Ratio",
    f"{results['Tier1 Ratio (Stressed)']:.2%}",
    f"{results['Tier1 Ratio (Stressed)'] - results['Tier1 Ratio (Base)']:+.2%}",
)
col6.metric("∆NII", f"{results['∆NII (Stressed)']:,.2f} EUR")

# --- Comparison Chart ---
st.subheader("Before vs. After Stress")

ratio_metrics = ["LCR", "NSFR", "CET1 Ratio", "Tier1 Ratio"]
absolute_metrics = ["∆EVE", "∆NII"]

chart_data = pd.DataFrame(
    {
        "Metric": ratio_metrics + absolute_metrics,
        "Base": [
            results["LCR (Base)"],
            results["NSFR (Base)"],
            results["CET1 Ratio (Base)"],
            results["Tier1 Ratio (Base)"],
            results["∆EVE (Base)"],
            results["∆NII (Base)"],
        ],
        "Stressed": [
            results["LCR (Stressed)"],
            results["NSFR (Stressed)"],
            results["CET1 Ratio (Stressed)"],
            results["Tier1 Ratio (Stressed)"],
            results["∆EVE (Stressed)"],
            results["∆NII (Stressed)"],
        ],
    }
).melt(id_vars="Metric", var_name="Condition", value_name="Value")

# Convert CET1 / Tier1 decimal ratios into % for visual comparability with LCR / NSFR
percent_scaling = ["CET1 Ratio", "Tier1 Ratio", "NSFR", "LCR"]
chart_ratios = chart_data[chart_data["Metric"].isin(ratio_metrics)].copy()
chart_ratios["Value"] = chart_ratios.apply(
    lambda r: r["Value"] * 100 if r["Metric"] in percent_scaling else r["Value"], axis=1
)

fig1 = px.bar(
    chart_ratios,
    x="Metric",
    y="Value",
    color="Condition",
    barmode="group",
    text_auto=".2f",
    title="Regulatory Ratios: Before vs. After (%)",
)
fig1.update_yaxes(title="%")
st.plotly_chart(fig1, width='stretch')

# ∆EVE and ∆NII: dumbbell chart showing the journey from Base to Stressed.
# Each metric gets its own panel (different EUR magnitudes), with a hollow
# Base marker, a filled Stressed marker, and a sign-coloured stripe between
# them. The axis is scaled to the *data*, not to regulatory thresholds —
# under the synthetic book the PV01-approximation ∆EVE (a few thousand EUR)
# is dwarfed by the EBA threshold (~15% × Tier1 ≈ several MEUR), so plotting
# the threshold as a vline would collapse the dumbbell onto the origin.
# Instead the threshold and the headroom % go in the subplot title and a
# caption underneath.
cap_ref = compute.calculate_capital_ratios(scenario_id=scenario_id)
tier1_capital = cap_ref["Tier1 Ratio"] * cap_ref["RWA"]
eba_threshold = 0.15 * tier1_capital  # |∆EVE| ≤ 15% × T1, supervisory outlier test


def _fmt_eur(v: float) -> str:
    av = abs(v)
    if av >= 1e9:
        return f"{v / 1e9:,.2f} B EUR"
    if av >= 1e6:
        return f"{v / 1e6:,.2f} M EUR"
    if av >= 1e3:
        return f"{v / 1e3:,.1f} k EUR"
    return f"{v:,.0f} EUR"


# Build subplot titles upfront — for ∆EVE we want the EBA context inline so
# the reader doesn't have to hunt for it elsewhere.
panel_titles = []
for m in absolute_metrics:
    if m == "∆EVE" and eba_threshold > 0:
        stressed_eve = float(results["∆EVE (Stressed)"])
        ratio = abs(stressed_eve) / eba_threshold if eba_threshold else 0.0
        headroom = max(0.0, 1.0 - ratio)
        breach = ratio > 1.0
        flag = " — BREACH" if breach else f" — headroom {headroom:.1%}"
        panel_titles.append(
            f"{m} — Base → Stressed (EBA threshold ±{_fmt_eur(eba_threshold)}{flag})"
        )
    else:
        panel_titles.append(f"{m} — Base → Stressed")

fig2 = make_subplots(
    rows=len(absolute_metrics), cols=1,
    shared_xaxes=False,
    vertical_spacing=0.35,
    subplot_titles=panel_titles,
)
for i, metric in enumerate(absolute_metrics, start=1):
    base_val = float(results[f"{metric} (Base)"])
    stressed_val = float(results[f"{metric} (Stressed)"])
    delta = stressed_val - base_val
    worse = abs(stressed_val) > abs(base_val)
    # Amber / teal palette — colorblind-friendlier than red / green and more
    # in keeping with the rest of the dashboard.
    arrow_color = "#d97706" if worse else "#0d9488"

    fig2.add_trace(
        go.Scatter(
            x=[base_val, stressed_val], y=[0, 0],
            mode="lines",
            line=dict(color=arrow_color, width=6),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=i, col=1,
    )
    fig2.add_trace(
        go.Scatter(
            x=[base_val, stressed_val], y=[0, 0],
            mode="markers+text",
            marker=dict(
                size=[18, 24],
                color=["white", "#334155"],
                line=dict(width=[3, 3], color=["#475569", "#0f172a"]),
                symbol=["circle", "circle"],
            ),
            text=[f"Base<br>{_fmt_eur(base_val)}", f"Stressed<br>{_fmt_eur(stressed_val)}"],
            textposition=["top center", "bottom center"],
            textfont=dict(size=11),
            showlegend=False,
            hovertemplate="%{text}<extra></extra>",
        ),
        row=i, col=1,
    )
    fig2.add_annotation(
        x=(base_val + stressed_val) / 2, y=0.55,
        text=f"<b>Δ stress: {_fmt_eur(delta)}</b>",
        showarrow=False,
        bgcolor=arrow_color,
        font=dict(color="white", size=12),
        borderpad=5,
        row=i, col=1,
    )
    fig2.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1, row=i, col=1)

    # Axis range driven by the data (NOT the EBA threshold) so the dumbbell
    # is always visible regardless of how small ∆EVE is relative to Tier1.
    data_span = max(abs(base_val), abs(stressed_val), abs(delta))
    if data_span == 0:
        data_span = 1.0
    fig2.update_xaxes(
        title_text="EUR", tickformat=",.0f",
        range=[-data_span * 1.6, data_span * 1.6],
        row=i, col=1,
    )
    fig2.update_yaxes(visible=False, range=[-1.2, 1.2], row=i, col=1)

fig2.update_layout(
    height=460,
    margin=dict(t=60, b=40, l=40, r=40),
    plot_bgcolor="white",
)
st.plotly_chart(fig2, width='stretch')
st.caption(
    "Each dumbbell shows where the metric sits at baseline (hollow circle) and "
    "where the stress moves it (filled slate circle). The stripe is **amber** "
    "when the stress pushes the metric further from zero (the loss direction) "
    "and **teal** when it brings it closer. The ∆EVE used here is the PV01-linear "
    "approximation (Total PV01 × shock) — for the full curve-revaluation EVE "
    "and the supervisory test, see the IRRBB page."
)
