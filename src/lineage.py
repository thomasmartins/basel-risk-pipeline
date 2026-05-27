"""Shared model-lineage expander used by every dashboard page.

Reads `mart_model_metadata` and renders the calibrated short-rate model
(family, params, half-life, curve-fit residual), the NMD overlay summary,
the CPR config, and a one-line digest of the LCR liquidity stress presets.
Each page calls `render_model_lineage()` near the top so a reviewer always
knows which engine version produced the numbers on screen.
"""

from __future__ import annotations

import json

import streamlit as st

from src import queries


_RUN_HINT = (
    "No risk-engine metadata found. Run `scripts/risk_engine.cmd` (or the "
    "Dagster `risk_engine_run` asset) to populate the marts."
)


def render_model_lineage(expanded: bool = False) -> None:
    """Drop the model-lineage expander onto the current Streamlit page."""
    with st.expander("Model lineage", expanded=expanded):
        meta = queries.get_mart("mart_model_metadata")
        if meta.empty:
            st.info(_RUN_HINT)
            return

        m = meta.iloc[0].to_dict()
        params = json.loads(m["params_json"])
        family = m["model_family"]
        display_name = {
            "hull_white_1f": "Hull-White 1F (curve-calibrated)",
            "vasicek_1f": "Vasicek 1F",
        }.get(family, family)

        c1, c2, c3 = st.columns(3)
        c1.metric("Short-rate model", f"{display_name} v{m['model_version']}")
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

        # NMD overlay
        nmd = json.loads(m["nmd_params_json"])
        st.caption(
            f"NMD overlay — core: {nmd['stable_core_pct']:.0%} @ "
            f"{nmd['core_behavioral_maturity_yrs']:.1f}y · "
            f"deposit β: {nmd['deposit_beta']:.2f}"
        )

        # CPR (added in Phase 2.1c)
        if "cpr_params_json" in m and m["cpr_params_json"]:
            cpr = json.loads(m["cpr_params_json"])
            st.caption(
                f"Mortgage CPR — base: {cpr['cpr_base']:.0%} · "
                f"β: {cpr['beta']:.1f} · cap: {cpr['cpr_cap']:.0%}"
            )

        # Liquidity stress presets (added in Phase 2.1d)
        if "liquidity_stress_params_json" in m and m["liquidity_stress_params_json"]:
            ls = json.loads(m["liquidity_stress_params_json"])
            for name in ("idiosyncratic", "market_wide", "combined"):
                if name in ls:
                    p = ls[name]
                    st.caption(
                        f"ALMM stress — {name}: retail "
                        f"{p['retail_stable_runoff']:.0%}/{p['retail_unstable_runoff']:.0%} · "
                        f"wholesale {p['wholesale_runoff']:.0%} · "
                        f"HQLA L2A/L2B haircuts {p['hqla_haircut_l2a']:.0%}/{p['hqla_haircut_l2b']:.0%}"
                    )
