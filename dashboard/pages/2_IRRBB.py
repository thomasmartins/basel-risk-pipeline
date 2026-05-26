import json

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import compute, queries
from src.scenario import scenario_sidebar

st.set_page_config(page_title="IRRBB", layout="wide")
st.title("Interest Rate Risk in the Banking Book (IRRBB)")

scenario_id = scenario_sidebar()
scenario_name = st.session_state.get("scenario_choice", "Baseline")

# ==========================================================
# Model lineage
# ==========================================================
with st.expander("Model lineage", expanded=False):
    meta = queries.get_mart("mart_model_metadata")
    if meta.empty:
        st.info(
            "No risk-engine metadata found. Run `python -m basel_risk_engine.run` to "
            "calibrate the short-rate model and produce IRRBB outputs."
        )
    else:
        m = meta.iloc[0].to_dict()
        params = json.loads(m["params_json"])
        family = m["model_family"]
        display_name = {
            "hull_white_1f": "Hull-White 1F (curve-calibrated)",
            "vasicek_1f": "Vasicek 1F",
        }.get(family, family)

        c1, c2, c3 = st.columns(3)
        c1.metric("Model", f"{display_name} v{m['model_version']}")
        c1.caption(f"Calibrated {m['calibration_timestamp'][:19]} UTC")

        if family == "hull_white_1f":
            c2.metric("a (mean reversion)", f"{params['a']:.3f}")
            c2.metric("Half-life", f"{m['half_life_years']:.2f} y")
            c3.metric("σ (volatility)", f"{params['sigma']:.3%}")
            c3.metric("Curve-fit residual", f"{m['curve_fit_max_residual']:.2e}")
        elif family == "vasicek_1f":
            c2.metric("κ (mean reversion)", f"{params['kappa']:.3f}")
            c2.metric("θ (long-run mean)", f"{params['theta']:.3%}")
            c3.metric("σ (volatility)", f"{params['sigma']:.3%}")
            c3.metric("Half-life", f"{m['half_life_years']:.2f} y")
            st.caption(
                f"Max |P_model(0,τ) − P_market(0,τ)| over curve tenor grid: "
                f"{m['curve_fit_max_residual']:.2e} "
                "(Vasicek does not pin to the observed curve — switch to Hull-White for arb-free pricing)."
            )

        st.caption(
            f"{int(m['calibration_n_obs'])} obs at Δt={m['calibration_dt']:.3f}y · "
            f"MC: {int(m['n_mc_paths'])} paths × {m['mc_horizon_years']:.1f}y."
        )
        nmd = json.loads(m["nmd_params_json"])
        st.caption(
            f"NMD overlay — core: {nmd['stable_core_pct']:.0%} @ "
            f"{nmd['core_behavioral_maturity_yrs']:.1f}y · "
            f"deposit β: {nmd['deposit_beta']:.2f}"
        )

# ==========================================================
# Risk Summary
# ==========================================================
st.markdown("### Risk Summary")

sup_df = queries.get_mart("mart_eve_supervisory", scenario_id=scenario_id)
bcbs_df = queries.get_mart("mart_eve_bcbs368", scenario_id=scenario_id)
dist_stats = queries.get_mart("mart_eve_distribution_stats", scenario_id=scenario_id)

if not sup_df.empty:
    sup = sup_df.iloc[0].to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Worst BCBS scenario", sup["worst_scenario"])
    c2.metric("Worst ∆EVE", f"{sup['worst_delta_eve']:,.0f} EUR")
    c3.metric("Tier 1", f"{sup['tier1_capital']:,.0f} EUR")
    c4.metric(
        "|∆EVE| / Tier1",
        f"{sup['ratio']:.2%}",
        delta="BREACH (>15%)" if sup["breach"] else "OK",
        delta_color="inverse" if sup["breach"] else "normal",
    )
    if sup.get("distributional_99"):
        st.caption(
            f"MC distributional |∆EVE|₉₉ at 1y horizon: "
            f"{sup['distributional_99']:,.0f} EUR (smaller than the deterministic "
            "shock because BCBS 368 applies an instantaneous parallel shift)."
        )
else:
    st.info("Run `python -m basel_risk_engine.run` to populate Phase 2 marts.")

# ==========================================================
# BCBS 368 six prescribed scenarios
# ==========================================================
st.subheader("∆EVE under BCBS 368 §132 scenarios")
if not bcbs_df.empty:
    bcbs_df = bcbs_df.sort_values("delta_eve")
    fig = px.bar(
        bcbs_df,
        x="shock_scenario",
        y="delta_eve",
        text_auto=".2s",
        color="delta_eve",
        color_continuous_scale="RdYlGn_r",
        labels={"delta_eve": "∆EVE (EUR)", "shock_scenario": "BCBS 368 scenario"},
    )
    fig.update_layout(coloraxis_showscale=False, height=380)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No BCBS 368 results yet.")

# ==========================================================
# MC ∆EVE distribution
# ==========================================================
st.subheader("MC ∆EVE distribution (1y forward horizon)")
dist_paths = queries.get_mart("mart_eve_distribution_stats", scenario_id=scenario_id)
dist_full = queries.get_mart("stg_risk_eve_distribution", scenario_id=scenario_id)
if not dist_full.empty:
    fig = px.histogram(
        dist_full, x="delta_eve", nbins=60,
        labels={"delta_eve": "∆EVE (EUR)"},
    )
    fig.update_layout(height=380, bargap=0.05, showlegend=False)
    if not dist_paths.empty:
        s = dist_paths.iloc[0]
        for label, value in [("p1", s["p01"]), ("p99", s["p99"]), ("mean", s["mean"])]:
            fig.add_vline(
                x=value, line_dash="dash",
                annotation_text=f"{label}={value:,.0f}",
                annotation_position="top",
            )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# PV01 Profile (still useful for shape inspection)
# ==========================================================
st.subheader("PV01 by tenor bucket")
pv01_df = compute.calculate_pv01_profile(scenario_id=scenario_id)
fig = px.bar(
    pv01_df, x="tenor_bucket", y="pv01",
    color="tenor_bucket", text_auto=".2f",
    labels={"pv01": "PV01 (EUR / bp)", "tenor_bucket": "Maturity bucket"},
)
fig.update_layout(showlegend=False, height=340)
st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FTP curve + NII attribution
# ==========================================================
st.subheader("FTP curve & NII attribution")

ftp_df = queries.get_mart("mart_ftp_curve")
attr_df = queries.get_mart("mart_nii_attribution", scenario_id=scenario_id)
attr_by_product_df = queries.get_mart("mart_nii_attribution_by_product", scenario_id=scenario_id)

if ftp_df.empty or attr_df.empty:
    st.info("Run `python -m basel_risk_engine.run` to populate FTP attribution marts.")
else:
    c_left, c_right = st.columns([1, 1])

    with c_left:
        # FTP curve: base + LP add-on
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ftp_df["tenor_years"], y=ftp_df["base_yield"] * 100,
            mode="lines+markers", name="Wholesale (base)",
        ))
        fig.add_trace(go.Scatter(
            x=ftp_df["tenor_years"], y=ftp_df["ftp_yield"] * 100,
            mode="lines+markers", name="FTP (base + LP)",
        ))
        fig.update_layout(
            height=340,
            xaxis_title="Tenor (years)",
            yaxis_title="Yield (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Internal FTP curve = wholesale base curve + tenor-dependent liquidity premium. "
            "Treasury charges (assets) or credits (liabilities) business units at the FTP yield "
            "matched to each cashflow's behavioural maturity."
        )

    with c_right:
        # NII attribution waterfall
        a = attr_df.iloc[0]
        fig = go.Figure(go.Waterfall(
            x=["Customer margin", "Funding margin", "Total NII"],
            measure=["relative", "relative", "total"],
            y=[a["customer_margin"], a["funding_margin"], 0],
            text=[f"{a['customer_margin']:,.0f}",
                  f"{a['funding_margin']:,.0f}",
                  f"{a['nii_total']:,.0f}"],
            textposition="outside",
        ))
        fig.update_layout(height=340, yaxis_title="EUR")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Behavioural value: **{a['behavioral_value']:,.0f} EUR** "
            "(slice of customer_margin credited to deposit business by pricing NMDs at "
            "behavioural maturity rather than contractual O/N — positive under an upward "
            "curve, which is the deposit unit's reward for sticky funding)."
        )

    if not attr_by_product_df.empty:
        st.markdown("**Per-product NII contribution**")
        fig = px.bar(
            attr_by_product_df,
            x="product",
            y=["customer_margin", "funding_margin"],
            barmode="stack",
            labels={"value": "NII (EUR)", "product": "Product", "variable": "Component"},
        )
        fig.update_layout(height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# MC ∆NII distribution
# ==========================================================
st.subheader("MC ∆NII paths (by horizon)")
nii_stats = queries.get_mart("mart_nii_horizon_stats", scenario_id=scenario_id)
nii_full = queries.get_mart("stg_risk_nii_paths", scenario_id=scenario_id)
if not nii_stats.empty:
    cols = st.columns(len(nii_stats))
    for i, row in nii_stats.sort_values("horizon_months").iterrows():
        with cols[i if i < len(cols) else 0]:
            st.metric(
                f"{int(row['horizon_months'])}m horizon",
                f"{row['mean']:,.0f} EUR",
                delta=f"p5..p95: [{row['p05']:,.0f}, {row['p95']:,.0f}]",
            )
if not nii_full.empty:
    nii_full["horizon"] = nii_full["horizon_months"].astype(str) + "m"
    fig = px.violin(
        nii_full, x="horizon", y="delta_nii", box=True, points=False,
        labels={"delta_nii": "∆NII over horizon (EUR)"},
    )
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# Parallel shock sensitivity (legacy, still useful)
# ==========================================================
st.subheader("Parallel shock ∆EVE sensitivity (PV01-approximation)")
shock_bps = st.slider(
    "Select parallel interest rate shock (bps)",
    min_value=-300, max_value=300, value=0, step=25, key="eve_shock_slider",
)
sensitivity = compute.calculate_eve_sensitivity(shock_bps=shock_bps, scenario_id=scenario_id)
c1, c2, c3 = st.columns(3)
c1.metric("Shock (bps)", sensitivity["Shock (bps)"])
c2.metric("Total PV01", f"{sensitivity['Total PV01']:,.2f} EUR")
c3.metric("∆EVE", f"{sensitivity['Delta EVE']:,.2f} EUR")
st.caption(
    "Linear PV01 × shock — useful for intuition, but the BCBS 368 panel above "
    "is the full curve-shift revaluation."
)
