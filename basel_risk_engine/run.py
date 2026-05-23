"""End-to-end risk-engine runner.

Pipeline:
    1. read short_rate_history -> calibrate Vasicek
    2. read yield_curve         -> seed the base curve
    3. simulate MC short-rate paths (5y horizon, monthly)
    4. for each scenario:
        - read cashflows + balance_sheet from DuckDB
        - apply NMD behavioural overlay
        - compute BCBS 368 deterministic ΔEVE
        - compute MC ΔEVE distribution
        - compute supervisory outlier test
        - compute NII paths under MC + NMD
    5. write Parquet outputs to data/risk_outputs/

CLI: `python -m basel_risk_engine.run [--n-paths 2000] [--horizon-years 5]`
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from basel_common.connection import warehouse_path
from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
from basel_risk_engine.rate_models import VasicekModel, simulate_paths
from basel_risk_engine.valuation.curve import YieldCurve
from basel_risk_engine.valuation.eve import EVEEngine
from basel_risk_engine.valuation.nii import compute_nii_paths

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "data" / "risk_outputs"

MODEL_NAME = "vasicek_1f"
MODEL_VERSION = "0.1.0"


def _read(con: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    return con.execute(sql, params or []).fetchdf()


def _build_base_curve(yc: pd.DataFrame) -> YieldCurve:
    yc_sorted = yc.sort_values("tenor_years")
    return YieldCurve(
        tenors_years=yc_sorted["tenor_years"].to_numpy(dtype=np.float64),
        zero_yields=yc_sorted["zero_yield"].to_numpy(dtype=np.float64),
    )


def run(
    out_dir: Path = _DEFAULT_OUT,
    *,
    n_paths: int = 2000,
    horizon_years: float = 5.0,
    dt: float = 1 / 12,
    seed: int = 7,
    nmd: NMDParams | None = None,
) -> dict[str, Path]:
    nmd = nmd or NMDParams()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(warehouse_path()), read_only=True)
    try:
        # --------------------------------- calibration
        history = _read(con, "SELECT short_rate FROM short_rate_history ORDER BY observation_date")
        calib_dt = 1 / 12
        calibration = VasicekModel.calibrate(history["short_rate"].to_numpy(), dt=calib_dt)
        model = VasicekModel(calibration.params)

        # --------------------------------- base curve
        yc = _read(con, "SELECT * FROM yield_curve")
        base_curve = _build_base_curve(yc)

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

        for sid in scenarios:
            cf = _read(
                con,
                """
                SELECT cashflow_id, product, amount, maturity_days
                FROM int_cashflows_enriched
                WHERE scenario_id = ?
                """,
                [sid],
            )
            if cf.empty:
                continue

            cf_b = apply_nmd_overlay(cf, nmd)

            engine = EVEEngine(base_curve=base_curve, vasicek_model=model)

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
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "calibration_timestamp": datetime.utcnow().isoformat(),
            "kappa": calibration.params.kappa,
            "theta": calibration.params.theta,
            "sigma": calibration.params.sigma,
            "r0": calibration.params.r0,
            "half_life_years": calibration.half_life_years,
            "n_calibration_obs": calibration.n_obs,
            "calibration_dt": calibration.dt,
            "n_mc_paths": n_paths,
            "mc_horizon_years": horizon_years,
            "mc_dt": dt,
            "nmd_params_json": json.dumps(nmd.model_dump()),
        }])
        _write("risk_model_metadata", meta_df)

        return written
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--horizon-years", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(f"Running risk engine -> {args.out.relative_to(_REPO_ROOT)}")
    run(args.out, n_paths=args.n_paths, horizon_years=args.horizon_years, seed=args.seed)


if __name__ == "__main__":
    main()
