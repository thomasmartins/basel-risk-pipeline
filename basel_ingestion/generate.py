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


# Per-product commercial spread over the wholesale base curve (annualised).
# Customer rate = base_yield(τ) + spread + jitter.
_PRODUCT_SPREAD = {
    "loan":    0.0200,   # +200bps over wholesale (commercial lending margin)
    "bond":    0.0050,   # +50bps over wholesale (credit risk on bonds we hold)
    "deposit": -0.0150,  # -150bps below wholesale (the bank pays less than wholesale to depositors)
}


def generate_cashflows(
    rng: np.random.Generator,
    dates: list[date],
    n: int,
    *,
    base_yield_fn,
    mortgage_share_of_loans: float = 0.30,
    callable_share_of_long_bonds: float = 0.40,
) -> pl.DataFrame:
    """Synthetic cashflow rows. Schema is extended for Phase 2.1c:

    - `amortization_type`        bullet|level. "level" rows are level-payment
                                 mortgages with explicit term_months; everything
                                 else is a bullet repaid at maturity.
    - `term_months`              months from origination to final maturity.
                                 Always populated (== maturity_days/30.4375 for
                                 bullets), but only consumed by the risk engine
                                 when amortization_type == 'level'.
    - `is_callable`, `call_date`, `call_strike_pct`
                                 European-call optionality on a subset of bonds.
                                 call_strike_pct is in % of par (100.0 = par).
                                 None/NaN for non-callable rows.

    Maturity ranges differ by product so the mortgage / long-bond subsets are
    realistic in tenor (5–30y for mortgages, 1–30y for bonds), and short-end
    deposits remain in the NMD-overlay-triggering range.
    """
    base_dates = rng.choice(dates, n)
    products = rng.choice(_enum_values(Product), n)
    products_arr = np.asarray(products)

    is_loan = products_arr == "loan"
    is_bond = products_arr == "bond"
    is_deposit = products_arr == "deposit"

    # Mortgage flag: a fraction of loans get level amortisation + long tenor.
    u_mortgage = rng.uniform(size=n)
    mortgage_mask = is_loan & (u_mortgage < mortgage_share_of_loans)

    # Per-product maturity offsets (days). Generated in one shot then patched.
    maturity_offsets = np.empty(n, dtype=np.int64)
    maturity_offsets[mortgage_mask] = rng.integers(5 * 365, 30 * 365 + 1, mortgage_mask.sum())
    short_loan_mask = is_loan & ~mortgage_mask
    maturity_offsets[short_loan_mask] = rng.integers(30, 365, short_loan_mask.sum())
    maturity_offsets[is_bond] = rng.integers(365, 30 * 365 + 1, is_bond.sum())
    maturity_offsets[is_deposit] = rng.integers(30, 365, is_deposit.sum())

    maturity_dates = [d + timedelta(days=int(off)) for d, off in zip(base_dates, maturity_offsets)]

    tenor_yrs = maturity_offsets.astype(np.float64) / 365.0
    base_yields = base_yield_fn(tenor_yrs)
    spreads = np.array([_PRODUCT_SPREAD.get(p, 0.0) for p in products_arr])
    jitter = rng.normal(0.0, 0.0010, n)  # ±10bps idiosyncratic noise
    customer_rates = base_yields + spreads + jitter

    # Amortisation
    amortization_type = np.where(mortgage_mask, "level", "bullet")
    term_months = np.maximum(1, (maturity_offsets / 30.4375).round().astype(np.int64))

    # Callable bonds: a subset of long-dated bonds get a half-life European call.
    # Strikes are randomised in [75, 88]% of par so the call has meaningful
    # value under our zero-coupon bond pricing. Real callable bonds carry coupon
    # streams that make them trade well above par, so a par-strike call is in
    # the money by approximately the cumulative coupon spread; setting the
    # strike below par here is the simplest equivalent of that economics under
    # the no-coupon synthetic model.
    u_call = rng.uniform(size=n)
    callable_mask = is_bond & (maturity_offsets > 5 * 365) & (u_call < callable_share_of_long_bonds)
    call_offset_days = (maturity_offsets * 0.5).astype(np.int64)
    call_dates: list[date | None] = [
        (d + timedelta(days=int(off))) if c else None
        for d, off, c in zip(base_dates, call_offset_days, callable_mask)
    ]
    call_strike_pct_raw = rng.uniform(75.0, 88.0, n)
    call_strike_pct = np.where(callable_mask, call_strike_pct_raw, np.nan)

    return pl.DataFrame(
        {
            "id": np.arange(1, n + 1, dtype=np.int64),
            "date": base_dates.tolist(),
            "product": products_arr.tolist(),
            "counterparty": rng.choice(_enum_values(Counterparty), n).tolist(),
            "maturity_date": maturity_dates,
            "bucket": rng.choice(["7d", "30d", "90d", "180d"], n).tolist(),
            "amount": rng.integers(10_000, 500_000, n).astype(np.float64).tolist(),
            "direction": rng.choice(_enum_values(Direction), n).tolist(),
            "hqlatype": rng.choice(
                _enum_values(HQLAType),
                n,
                p=[0.15, 0.05, 0.05, 0.75],  # Level1, Level2A, Level2B, None
            ).tolist(),
            # ASF / RSF distributions calibrated so NSFR sits ~1.20, matching a
            # well-funded EU bank under CRR II (typical reported range 105-135%).
            #   E[asf] = 0·0.10 + 0.5·0.20 + 0.95·0.60 + 1.0·0.10 = 0.77
            #   E[rsf] = 0.05·0.20 + 0.65·0.40 + 0.85·0.25 + 1.0·0.15 = 0.6325
            #   NSFR  ≈ 0.77 / 0.6325 ≈ 1.22
            "asf_factor": rng.choice(
                [0.0, 0.5, 0.95, 1.0], n, p=[0.10, 0.20, 0.60, 0.10]
            ).tolist(),
            "rsf_factor": rng.choice(
                [0.05, 0.65, 0.85, 1.0], n, p=[0.20, 0.40, 0.25, 0.15]
            ).tolist(),
            "customer_rate": customer_rates.tolist(),
            "amortization_type": amortization_type.tolist(),
            "term_months": term_months.tolist(),
            "is_callable": callable_mask.tolist(),
            "call_date": call_dates,
            "call_strike_pct": call_strike_pct.tolist(),
            "scenario_id": rng.integers(1, 5, n, dtype=np.int64).tolist(),
        },
        schema_overrides={"call_date": pl.Date},
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


def generate_balance_sheet(rng: np.random.Generator, dates: list[date]) -> pl.DataFrame:
    """One row per (scenario × date × balance-sheet item), with a coherent
    capital stack at every snapshot.

    Sized against the RWA generator: per (date × scenario) RWA totals are
    ~3.4M EUR after the n_rwa=5000 calibration, so capital levels are picked
    so CET1 / Tier1 / Total ratios land in the realistic 11-17 % band, and
    the stack invariant CET1 ≤ Tier1 ≤ Total Capital is enforced by
    construction (Tier1 = CET1 + AT1, Total = Tier1 + Tier2 with positive
    increments).
    """
    n_scenarios = 4
    rows = []
    next_id = 1
    for scenario_id in range(1, n_scenarios + 1):
        for d in dates:
            cet1 = float(rng.uniform(380_000, 470_000))         # ~ 12-14 % of RWA
            at1 = float(rng.uniform(40_000, 75_000))            # AT1 sliver
            tier1 = cet1 + at1                                  # 14-16 %
            tier2 = float(rng.uniform(70_000, 130_000))         # Tier 2
            total_capital = tier1 + tier2                       # 17-19 %
            total_assets = float(rng.uniform(40_000_000, 60_000_000))
            total_liabilities = total_assets - total_capital
            for item, amount in [
                ("CET1", cet1),
                ("Tier1", tier1),
                ("Total Capital", total_capital),
                ("Total Assets", total_assets),
                ("Total Liabilities", total_liabilities),
            ]:
                rows.append({
                    "id": next_id,
                    "date": d,
                    "item": item,
                    "amount": amount,
                    "scenario_id": scenario_id,
                })
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


def generate_liquidity_premium(valuation_date: date) -> pl.DataFrame:
    """Internal FTP liquidity-premium add-on, in basis points per tenor.

    Realistic shape: monotonically increasing from 0bps overnight to ~75bps at
    10y, flattening at the long end. Treasury charges the longer-tenor LP
    because illiquid funding (long-dated wholesale) commands a premium over
    the base wholesale curve.
    """
    tenors_years = [0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    tenor_labels = ["1m", "3m", "6m", "1y", "2y", "3y", "5y", "7y", "10y", "20y", "30y"]
    lp_bps = [2.0, 5.0, 10.0, 18.0, 30.0, 40.0, 55.0, 65.0, 75.0, 80.0, 80.0]
    return pl.DataFrame(
        {
            "valuation_date": [valuation_date] * len(tenors_years),
            "tenor_label": tenor_labels,
            "tenor_years": tenors_years,
            "lp_bps": lp_bps,
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
    n_rwa: int = 5_000,
    start: date = date(2024, 1, 1),
) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    dates = _date_range(start, periods)
    out_dir.mkdir(parents=True, exist_ok=True)

    short_rate_history, true_vasicek = generate_short_rate_history(rng, valuation_date=start)
    last_short_rate = float(short_rate_history["short_rate"][-1])
    yield_curve = generate_yield_curve(start, last_short_rate, true_vasicek)

    # Interpolant the cashflow generator uses to set customer rates per row.
    yc_tenors = yield_curve["tenor_years"].to_numpy()
    yc_yields = yield_curve["zero_yield"].to_numpy()
    base_yield_fn = lambda taus: np.interp(taus, yc_tenors, yc_yields)

    tables = {
        "scenarios": generate_scenarios(),
        "cashflows": generate_cashflows(rng, dates, n_cashflows, base_yield_fn=base_yield_fn),
        "rwa": generate_rwa(rng, dates, n_rwa),
        "balance_sheet": generate_balance_sheet(rng, dates),
        "params": generate_params(),
        "short_rate_history": short_rate_history,
        "yield_curve": yield_curve,
        "liquidity_premium": generate_liquidity_premium(start),
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
            n_rwa=200,
        )
    else:
        print(f"Generating FULL dataset at {args.out.relative_to(_REPO_ROOT)} (seed={args.seed})")
        generate_all(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
