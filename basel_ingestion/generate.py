"""Synthetic Parquet generator for the ALM warehouse.

Phase 0 keeps the row shapes of the original Postgres schema 1:1 — only the
storage layer changes. Phase 1+ extends shapes (market curves, behavioral
parameters, model_metadata) and partitions by valuation_date.

Run: `python -m basel_ingestion.generate [--out data/raw] [--small] [--seed 42]`
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from basel_common.types import (
    Approach,
    AssetClass,
    BalanceSheetItem,
    Counterparty,
    Direction,
    HQLAType,
    Product,
    TenorBucket,
)
from basel_risk_engine.rate_models import VasicekModel, VasicekParams

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OUT = _REPO_ROOT / "data" / "raw"


def _enum_values(enum_cls) -> list[str]:
    return [m.value for m in enum_cls]


def _date_range(start: date, periods: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(periods)]


def generate_scenarios() -> pl.DataFrame:
    """Fixed scenario table. IDs 1..4 are referenced by FK throughout."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Baseline", "ECB Stress", "Liquidity Shock", "Interest Rate Shock"],
            "description": [
                "Normal conditions",
                "Comprehensive ECB stress scenario",
                "30% wholesale funding withdrawal",
                "Parallel +200bps rate shift",
            ],
            "liquidity_shock": [0.0, 20.0, 30.0, 0.0],
            "ir_shift": [0.0, 100.0, 0.0, 200.0],
            "credit_shock": [0.0, 50.0, 0.0, 0.0],
        }
    )


def generate_cashflows(rng: np.random.Generator, dates: list[date], n: int) -> pl.DataFrame:
    base_dates = rng.choice(dates, n)
    maturity_offsets = rng.integers(30, 365, n)
    maturity_dates = [d + timedelta(days=int(off)) for d, off in zip(base_dates, maturity_offsets)]
    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int64),
            "date": base_dates.tolist(),
            "product": rng.choice(_enum_values(Product), n).tolist(),
            "counterparty": rng.choice(_enum_values(Counterparty), n).tolist(),
            "maturity_date": maturity_dates,
            "bucket": rng.choice(["7d", "30d", "90d", "180d"], n).tolist(),
            "amount": rng.integers(10_000, 500_000, n).astype(np.float64).tolist(),
            "direction": rng.choice(_enum_values(Direction), n).tolist(),
            "hqlatype": rng.choice(_enum_values(HQLAType), n).tolist(),
            "asf_factor": rng.choice([0.0, 0.5, 0.9], n).tolist(),
            "rsf_factor": rng.choice([0.05, 0.85, 1.0], n).tolist(),
            "scenario_id": rng.integers(1, 5, n, dtype=np.int64).tolist(),
        }
    )


def generate_rwa(rng: np.random.Generator, dates: list[date], n: int) -> pl.DataFrame:
    amount = rng.integers(50_000, 1_000_000, n).astype(np.float64)
    risk_weight = rng.choice([0.0, 0.35, 0.5, 1.0], n)
    rwa_amount = amount * risk_weight
    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int64),
            "date": rng.choice(dates, n).tolist(),
            "exposure_id": [f"EXP{i:04d}" for i in range(n)],
            "asset_class": rng.choice(_enum_values(AssetClass), n).tolist(),
            "approach": rng.choice(_enum_values(Approach), n).tolist(),
            "amount": amount.tolist(),
            "risk_weight": risk_weight.tolist(),
            "rwa_amount": rwa_amount.tolist(),
            "capital_requirement": (rwa_amount * 0.08).tolist(),
            "scenario_id": rng.integers(1, 5, n, dtype=np.int64).tolist(),
        }
    )


def generate_irrbb(rng: np.random.Generator, dates: list[date], n: int) -> pl.DataFrame:
    base_dates = rng.choice(dates, n)
    maturity_offsets = rng.integers(30, 3650, n)
    maturity_dates = [d + timedelta(days=int(off)) for d, off in zip(base_dates, maturity_offsets)]
    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int64),
            "date": base_dates.tolist(),
            "instrument": [f"INST{i:04d}" for i in range(n)],
            "cashflow": rng.integers(-100_000, 100_000, n).astype(np.float64).tolist(),
            "maturity_date": maturity_dates,
            "tenor_bucket": rng.choice(_enum_values(TenorBucket), n).tolist(),
            "pv01": rng.normal(0, 1, n).round(6).tolist(),
            "rate_sensitivity": rng.normal(0, 1, n).round(6).tolist(),
            "scenario_id": rng.integers(1, 5, n, dtype=np.int64).tolist(),
        }
    )


def generate_balance_sheet(rng: np.random.Generator, dates: list[date]) -> pl.DataFrame:
    items = _enum_values(BalanceSheetItem)
    rows = []
    next_id = 1
    for d in dates:
        for item in items:
            rows.append(
                {
                    "id": next_id,
                    "date": d,
                    "item": item,
                    "amount": float(rng.integers(1_000_000, 10_000_000)),
                    "scenario_id": int(rng.integers(1, 5)),
                }
            )
            next_id += 1
    return pl.DataFrame(rows)


def generate_short_rate_history(
    rng: np.random.Generator,
    *,
    valuation_date: date,
    n_years: int = 5,
    dt_months: int = 1,
    true_params: VasicekParams | None = None,
) -> tuple[pl.DataFrame, VasicekParams]:
    """Forward-simulate a synthetic short-rate history under a known Vasicek
    SDE. The 'true' parameters are also returned so calibration can be
    compared against ground truth during testing.
    """
    p = true_params or VasicekParams(kappa=0.5, theta=0.025, sigma=0.01, r0=0.03)
    n_steps = n_years * 12 // dt_months
    dt = dt_months / 12
    model = VasicekModel(p)
    sim = model.simulate(
        n_paths=1,
        n_steps=n_steps,
        dt=dt,
        seed=int(rng.integers(0, 2**31 - 1)),
        antithetic=False,
    )[0]
    dates = [
        date(valuation_date.year - n_years + (i * dt_months) // 12,
             ((i * dt_months) % 12) + 1, 1)
        for i in range(n_steps + 1)
    ]
    return (
        pl.DataFrame({"observation_date": dates, "short_rate": sim.tolist()}),
        p,
    )


def generate_yield_curve(
    valuation_date: date,
    short_rate: float,
    params: VasicekParams,
) -> pl.DataFrame:
    """Initial zero curve from the Vasicek analytical bond formula."""
    tenors_years = [0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    tenor_labels = ["1m", "3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
    model = VasicekModel(params)
    taus = np.array(tenors_years)
    discounts = model.bond_price(taus, short_rate)
    yields = model.zero_yield(taus, short_rate)
    return pl.DataFrame(
        {
            "valuation_date": [valuation_date] * len(tenors_years),
            "tenor_label": tenor_labels,
            "tenor_years": tenors_years,
            "discount_factor": discounts.tolist(),
            "zero_yield": yields.tolist(),
        }
    )


def generate_params() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": [
                "asf_factor_retail_stable",
                "asf_factor_retail_less_stable",
                "asf_factor_wholesale_lt1y",
                "asf_factor_wholesale_gt1y",
                "rsf_factor_loans_gt1y",
                "rsf_factor_loans_lt1y",
                "rsf_factor_hqla_level1",
                "rsf_factor_other_assets",
                "lcr_inflow_cap",
                "lcr_outflow_cap",
                "haircut_level2a",
                "haircut_level2b",
                "eve_tier1_breach_ratio",
                "capital_requirement_ratio",
            ],
            "value": [
                "0.95",
                "0.90",
                "0.0",
                "0.5",
                "1.0",
                "0.85",
                "0.05",
                "1.0",
                "0.75",
                "1.00",
                "0.15",
                "0.50",
                "0.15",
                "0.08",
            ],
        }
    )


def generate_all(
    out_dir: Path,
    *,
    seed: int = 42,
    periods: int = 90,
    n_cashflows: int = 5_000,
    n_rwa: int = 1_000,
    n_irrbb: int = 500,
    start: date = date(2024, 1, 1),
) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    dates = _date_range(start, periods)
    out_dir.mkdir(parents=True, exist_ok=True)

    short_rate_history, true_vasicek = generate_short_rate_history(rng, valuation_date=start)
    last_short_rate = float(short_rate_history["short_rate"][-1])

    tables = {
        "scenarios": generate_scenarios(),
        "cashflows": generate_cashflows(rng, dates, n_cashflows),
        "rwa": generate_rwa(rng, dates, n_rwa),
        "irrbb": generate_irrbb(rng, dates, n_irrbb),
        "balance_sheet": generate_balance_sheet(rng, dates),
        "params": generate_params(),
        "short_rate_history": short_rate_history,
        "yield_curve": generate_yield_curve(start, last_short_rate, true_vasicek),
    }

    written = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.write_parquet(path)
        written[name] = path
        print(f"  wrote {name:15s} {df.height:>6d} rows -> {path.relative_to(_REPO_ROOT)}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--small",
        action="store_true",
        help="Generate seed-sized sample (for tests/CI); writes to data/seed by default",
    )
    args = parser.parse_args()

    if args.small:
        out = args.out if args.out != _DEFAULT_OUT else _REPO_ROOT / "data" / "seed"
        print(f"Generating SMALL sample at {out.relative_to(_REPO_ROOT)} (seed={args.seed})")
        generate_all(
            out,
            seed=args.seed,
            periods=30,
            n_cashflows=200,
            n_rwa=50,
            n_irrbb=30,
        )
    else:
        print(f"Generating FULL dataset at {args.out.relative_to(_REPO_ROOT)} (seed={args.seed})")
        generate_all(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
