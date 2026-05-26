"""End-to-end risk-engine runner.

Pipeline:
    1. read short_rate_history -> calibrate the chosen short-rate model
    2. read yield_curve         -> seed the base curve (and theta(t) for HW1F)
    3. simulate MC short-rate paths (5y horizon, monthly)
    4. for each scenario:
        - read cashflows + balance_sheet from DuckDB
        - apply NMD behavioural overlay
        - compute BCBS 368 deterministic ΔEVE
        - compute MC ΔEVE distribution
        - compute supervisory outlier test
        - compute NII paths under MC + NMD
    5. write Parquet outputs to data/risk_outputs/

Model choice:
    --model hull_white  (default) — arbitrage-free against the observed curve;
                                    theta(t) bootstrapped from forward curve.
    --model vasicek                — Phase 2 baseline; constant theta from history.

CLI: `python -m basel_risk_engine.run [--model hull_white] [--n-paths 2000] [--horizon-years 5]`
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from basel_common.connection import warehouse_path
from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
from basel_risk_engine.ftp import (
    FTPCurve,
    LiquidityPremiumSchedule,
    compute_attribution,
)
from basel_risk_engine.rate_models import (
    HullWhiteModel,
    VasicekModel,
    simulate_paths,
)
from basel_risk_engine.valuation.curve import YieldCurve
from basel_risk_engine.valuation.eve import EVEEngine
from basel_risk_engine.valuation.nii import compute_nii_paths

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "data" / "risk_outputs"

MODEL_VERSION = "0.2.0"
_AVAILABLE_MODELS = ("hull_white", "vasicek")


def _read(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).fetchdf()


def _build_base_curve(yc: pd.DataFrame) -> YieldCurve:
    yc_sorted = yc.sort_values("tenor_years")
    return YieldCurve(
        tenors_years=yc_sorted["tenor_years"].to_numpy(dtype=np.float64),
        zero_yields=yc_sorted["zero_yield"].to_numpy(dtype=np.float64),
    )


def _build_ftp_curve(base_curve: YieldCurve, lp: pd.DataFrame) -> FTPCurve:
    lp_sorted = lp.sort_values("tenor_years")
    schedule = LiquidityPremiumSchedule(
        tenors_years=lp_sorted["tenor_years"].to_numpy(dtype=np.float64),
        lp_bps=lp_sorted["lp_bps"].to_numpy(dtype=np.float64),
    )
    return FTPCurve(base_curve=base_curve, lp_schedule=schedule)


def _calibrate(model_name: str, history: np.ndarray, dt: float, base_curve: YieldCurve):
    """Dispatch calibration. Returns (model, model_family, params_dict, calib_info_dict)."""
    if model_name == "hull_white":
        # Pin r0 to the curve's instantaneous forward at t=0 so the model
        # reprices the input curve exactly (otherwise small drift from
        # history's final observation leaks into the curve-fit residual).
        r0_curve = float(base_curve.forward_rate(np.array([1e-12]))[0])
        calib = HullWhiteModel.calibrate(history, dt=dt, market_curve=base_curve, r0_override=r0_curve)
        model = HullWhiteModel(calib.params, base_curve)
        info = {
            "n_obs": calib.n_obs,
            "dt": calib.dt,
            "half_life_years": calib.half_life_years,
            "log_likelihood": calib.log_likelihood,
            "curve_fit_max_residual": calib.curve_fit_max_residual,
        }
        return model, "hull_white_1f", calib.params.model_dump(), info

    if model_name == "vasicek":
        calib = VasicekModel.calibrate(history, dt=dt)
        model = VasicekModel(calib.params)
        # Vasicek's term structure does not in general match P^M; measure the gap.
        tenor_grid = base_curve.tenors_years
        residual = float(
            np.max(np.abs(model.bond_price(tenor_grid, calib.params.r0) - base_curve.discount_factor(tenor_grid)))
        )
        info = {
            "n_obs": calib.n_obs,
            "dt": calib.dt,
            "half_life_years": calib.half_life_years,
            "log_likelihood": calib.log_likelihood,
            "curve_fit_max_residual": residual,
        }
        return model, "vasicek_1f", calib.params.model_dump(), info

    raise ValueError(f"Unknown model {model_name!r}; available: {_AVAILABLE_MODELS}")


def run(
    out_dir: Path = _DEFAULT_OUT,
    *,
    model_name: str = "hull_white",
    n_paths: int = 2000,
    horizon_years: float = 5.0,
    dt: float = 1 / 12,
    seed: int = 7,
    nmd: NMDParams | None = None,
) -> dict[str, Path]:
    if model_name not in _AVAILABLE_MODELS:
        raise ValueError(f"Unknown model {model_name!r}; available: {_AVAILABLE_MODELS}")
    nmd = nmd or NMDParams()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(warehouse_path()), read_only=True)
    try:
        # --------------------------------- base curve (needed for HW1F calibration)
        yc = _read(con, "SELECT * FROM yield_curve")
        base_curve = _build_base_curve(yc)

        # --------------------------------- FTP curve = base + liquidity premium
        lp = _read(con, "SELECT * FROM liquidity_premium")
        ftp_curve = _build_ftp_curve(base_curve, lp)

        # --------------------------------- model calibration
        history = _read(con, "SELECT short_rate FROM short_rate_history ORDER BY observation_date")
        calib_dt = 1 / 12
        model, model_family, params_dict, calib_info = _calibrate(
            model_name, history["short_rate"].to_numpy(), calib_dt, base_curve
        )

        # --------------------------------- MC paths (market-wide, scenario-agnostic)
        paths = simulate_paths(
            model,
            n_paths=n_paths,
            horizon_years=horizon_years,
            dt=dt,
            seed=seed,
            antithetic=True,
        )

        scenarios = _read(con, "SELECT id FROM scenarios ORDER BY id")["id"].tolist()

        # --------------------------------- per-scenario valuation
        eve_supervisory_rows: list[dict] = []
        eve_distribution_rows: list[dict] = []
        nii_rows: list[dict] = []
        bcbs_rows: list[dict] = []
        attribution_rows: list[dict] = []
        attribution_book_rows: list[dict] = []

        for sid in scenarios:
            cf = _read(
                con,
                """
                SELECT cashflow_id, product, amount, maturity_days, customer_rate
                FROM int_cashflows_enriched
                WHERE scenario_id = ?
                """,
                [sid],
            )
            if cf.empty:
                continue

            cf_b = apply_nmd_overlay(cf, nmd)

            engine = EVEEngine(base_curve=base_curve, rate_model=model)

            # BCBS 368 deterministic
            bcbs = engine.bcbs368(cf_b)
            for r in bcbs:
                bcbs_rows.append({
                    "scenario_id": sid, "shock_scenario": r.scenario, "delta_eve": r.delta_eve,
                })

            # Tier1 capital for this scenario
            tier1 = _read(
                con,
                "SELECT tier1 FROM mart_capital_ratios WHERE scenario_id = ?",
                [sid],
            )
            tier1_value = float(tier1.iloc[0, 0]) if not tier1.empty else 0.0

            # MC distribution
            mc_dist = engine.mc_distribution(cf_b, paths, forward_horizon_years=1.0)
            for p_id, val in enumerate(mc_dist):
                eve_distribution_rows.append({
                    "scenario_id": sid, "path_id": p_id, "delta_eve": float(val),
                })

            # Supervisory outlier test (with distributional p99)
            sup = engine.supervisory_outlier_test(
                cf_b, tier1_capital=tier1_value, distributional_paths=paths
            )
            eve_supervisory_rows.append({
                "scenario_id": sid,
                "worst_scenario": sup.worst_scenario,
                "worst_delta_eve": sup.worst_delta_eve,
                "tier1_capital": sup.tier1_capital,
                "ratio": sup.ratio,
                "breach": sup.breach,
                "distributional_99": sup.distributional_99,
            })

            # NII paths
            nii = compute_nii_paths(cf_b, paths, nmd=nmd)
            nii["scenario_id"] = sid
            nii_rows.extend(nii.to_dict("records"))

            # FTP attribution (static, baseline NMD overlay)
            attr = compute_attribution(cf_b, ftp_curve)
            per_row = attr.per_row.copy()
            per_row["scenario_id"] = sid
            attribution_rows.extend(per_row.to_dict("records"))
            book = attr.book_total
            attribution_book_rows.append({
                "scenario_id": sid,
                "customer_margin": float(book["customer_margin"]),
                "funding_margin": float(book["funding_margin"]),
                "behavioral_value": float(book["behavioral_value"]),
                "nii_total": float(book["nii_total"]),
            })

        # --------------------------------- write outputs
        written: dict[str, Path] = {}

        def _write(name: str, df: pd.DataFrame) -> None:
            path = out_dir / f"{name}.parquet"
            pl.from_pandas(df).write_parquet(path)
            written[name] = path
            print(f"  wrote {name:30s} {df.shape[0]:>7d} rows -> {path.relative_to(_REPO_ROOT)}")

        _write("risk_eve_bcbs368", pd.DataFrame(bcbs_rows))
        _write("risk_eve_supervisory", pd.DataFrame(eve_supervisory_rows))
        _write("risk_eve_distribution", pd.DataFrame(eve_distribution_rows))
        _write("risk_nii_paths", pd.DataFrame(nii_rows))
        _write("risk_nii_attribution", pd.DataFrame(attribution_book_rows))
        _write("risk_nii_attribution_rows", pd.DataFrame(attribution_rows))

        # FTP curve snapshot (one row per tenor): base, lp, total
        ftp_grid = ftp_curve.to_grid_frame()
        ftp_df = pd.DataFrame(ftp_grid)
        _write("risk_ftp_curve", ftp_df)

        # rate paths — downsample to keep file size sane
        keep_paths = min(200, n_paths)
        rate_path_df = pd.DataFrame({
            "path_id": np.repeat(np.arange(keep_paths), paths.n_steps + 1),
            "step": np.tile(np.arange(paths.n_steps + 1), keep_paths),
            "time_years": np.tile(np.arange(paths.n_steps + 1) * paths.dt, keep_paths),
            "short_rate": paths.short_rates[:keep_paths].ravel(),
        })
        _write("risk_rate_paths", rate_path_df)

        meta_df = pd.DataFrame([{
            "model_family": model_family,
            "model_version": MODEL_VERSION,
            "calibration_timestamp": datetime.now(timezone.utc).isoformat(),
            "params_json": json.dumps(params_dict),
            "calibration_n_obs": int(calib_info["n_obs"]),
            "calibration_dt": float(calib_info["dt"]),
            "half_life_years": float(calib_info["half_life_years"]),
            "log_likelihood": float(calib_info["log_likelihood"]),
            "curve_fit_max_residual": float(calib_info["curve_fit_max_residual"]),
            "n_mc_paths": int(n_paths),
            "mc_horizon_years": float(horizon_years),
            "mc_dt": float(dt),
            "nmd_params_json": json.dumps(nmd.model_dump()),
        }])
        _write("risk_model_metadata", meta_df)

        return written
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--model", choices=_AVAILABLE_MODELS, default="hull_white")
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--horizon-years", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"Running risk engine [{args.model}] -> {args.out.relative_to(_REPO_ROOT)}")
    run(
        args.out,
        model_name=args.model,
        n_paths=args.n_paths,
        horizon_years=args.horizon_years,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
