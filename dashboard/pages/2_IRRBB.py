import _bootstrap  # noqa: F401  -- must precede `src`/`basel_common` imports

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import compute, queries
from src.lineage import render_model_lineage
from src.scenario import scenario_sidebar

st.set_page_config(page_title="IRRBB", layout="wide")
st.title("Interest Rate Risk in the Banking Book (IRRBB)")

scenario_id = scenario_sidebar()
scenario_name = st.session_state.get("scenario_choice", "Baseline")

render_model_lineage(expanded=False)

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
    st.plotly_chart(fig, width='stretch')
else:
    st.info("No BCBS 368 results yet.")

# ==========================================================
# Optionality: mortgage CPR + callable bonds (Phase 2.1c)
# ==========================================================
st.subheader("Embedded optionality — mortgage CPR + callable bonds")

opt_df = queries.get_mart("mart_optionality_summary", scenario_id=scenario_id)
mort_df = queries.get_mart("mart_mortgage_cf", scenario_id=scenario_id)
cb_df = queries.get_mart("mart_callable_bonds", scenario_id=scenario_id)

if opt_df.empty:
    st.info("Run `python -m basel_risk_engine.run` to populate optionality marts.")
else:
    o = opt_df.iloc[0].to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mortgages", f"{int(o['n_mortgages'])}")
    c1.caption(f"Avg CPR: {o['mortgage_avg_cpr']:.1%} · WAL: {o['mortgage_avg_wal_years']:.2f}y")
    c2.metric(
        "Mortgage PV (CPR-adjusted)",
        f"{o['mortgage_pv_cpr']:,.0f} EUR",
        delta=f"{o['mortgage_pv_cpr_impact']:,.0f} vs scheduled",
        delta_color="inverse",
    )
    c3.metric("Callable bonds", f"{int(o['n_callable_bonds'])}")
    c3.caption(f"Avg integrated vol: {o['callable_avg_integrated_vol']:.4f}")
    c4.metric(
        "Total call value (option drag)",
        f"-{o['callable_call_value']:,.0f} EUR",
        delta=(
            f"{o['callable_call_value'] / o['callable_straight_pv']:.2%} of straight"
            if o["callable_straight_pv"] > 0 else None
        ),
        delta_color="inverse",
    )

    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("**Mortgage CPR vs WAL** (one dot per mortgage)")
        if not mort_df.empty:
            fig = px.scatter(
                mort_df,
                x="weighted_avg_life_years",
                y="avg_cpr",
                size="notional",
                color="contract_rate",
                color_continuous_scale="Viridis",
                hover_data=["cashflow_id", "term_months", "effective_term_months"],
                labels={
                    "weighted_avg_life_years": "Weighted-avg life (years)",
                    "avg_cpr": "Avg CPR",
                    "contract_rate": "Contract rate",
                },
            )
            fig.update_layout(height=340)
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Higher contract rates and lower market refi rates push CPR up, which shortens "
                "the mortgage's weighted-average life — the standard refi-incentive pattern."
            )
        else:
            st.info("No mortgages in this scenario.")

    with c_right:
        st.markdown("**Top 10 callable bonds — Black-76 decomposition**")
        if not cb_df.empty:
            top10 = cb_df.nlargest(10, "call_value").copy()
            top10["bond_id"] = "B" + top10["cashflow_id"].astype(str)
            top10["bond_pv_after_call"] = top10["callable_pv"]
            top10 = top10.melt(
                id_vars=["bond_id"],
                value_vars=["bond_pv_after_call", "call_value"],
                var_name="component",
                value_name="value",
            )
            top10["component"] = top10["component"].map({
                "bond_pv_after_call": "Callable PV (holder keeps)",
                "call_value": "Call value (issuer's option)",
            })
            fig = px.bar(
                top10,
                x="bond_id",
                y="value",
                color="component",
                barmode="stack",
                labels={"value": "EUR", "bond_id": "Bond"},
            )
            fig.update_layout(
                height=340,
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0),
            )
            st.plotly_chart(fig, width='stretch')
            st.caption(
                "Each bar's full height is the straight (non-callable) PV; the upper segment is "
                "the Black-76 call value the issuer holds — i.e. the discount the bondholder "
                "accepts for selling that optionality."
            )
        else:
            st.info("No callable bonds in this scenario.")

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
    st.plotly_chart(fig, width='stretch')

# ==========================================================
# PV01 Profile (still useful for shape inspection)
# ==========================================================
st.subheader("PV01 by tenor bucket")
pv01_df = compute.calculate_pv01_profile(scenario_id=scenario_id)
bucket_order = ["0-1y", "1-3y", "3-5y", "5-10y", "10y+"]
pv01_df = (
    pv01_df.set_index("tenor_bucket").reindex(bucket_order).fillna(0).reset_index()
)
total_abs = pv01_df["pv01"].abs().sum()
pv01_df["pct_of_total"] = (
    pv01_df["pv01"].abs() / total_abs * 100 if total_abs > 0 else 0
)
pv01_df["label"] = pv01_df.apply(
    lambda r: f"{r['pv01']:,.0f}  ({r['pct_of_total']:.1f}% of |total|)", axis=1
)
fig = go.Figure(
    go.Bar(
        x=pv01_df["pv01"],
        y=pv01_df["tenor_bucket"],
        orientation="h",
        text=pv01_df["label"],
        textposition="auto",
        marker_color=["#d62728" if v < 0 else "#1f77b4" for v in pv01_df["pv01"]],
    )
)
fig.add_vline(x=0, line_color="grey", line_width=1)
fig.update_layout(
    height=340,
    xaxis_title="PV01 (EUR per bp)",
    yaxis=dict(
        title="Maturity bucket",
        categoryorder="array",
        categoryarray=bucket_order[::-1],  # 0-1y on top, 10y+ at bottom
    ),
    showlegend=False,
    margin=dict(l=80, r=40),
)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Sign-coloured (red = negative / receiver position, blue = positive / payer). "
    "Per-EUR PV01 sensitivity scales with τ × DF(τ), so the 10y+ bucket dominates "
    "absolute PV01 even when notional is spread across shorter buckets."
)

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
        st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig, width='stretch')
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
        st.plotly_chart(fig, width='stretch')

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
    st.plotly_chart(fig, width='stretch')

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
