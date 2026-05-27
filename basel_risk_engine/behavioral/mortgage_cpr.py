"""Mortgage prepayment (CPR) and level-payment amortisation schedule.

For a level-payment fixed-rate mortgage with notional N, contract rate c
(annualised), and term n months, the monthly payment is

    P = N * (c/12) / (1 - (1 + c/12)^(-n))

The contractual schedule decomposes each P_t into

    I_t = (c/12) * B_{t-1}      (interest)
    A_t = P - I_t                (scheduled principal)
    B_t = B_{t-1} - A_t          (closing balance, B_0 = N)

so the bookkeeping identity sum(A_t) = N holds by construction.

CPR (Conditional Prepayment Rate) is the *annualised* fraction of the
remaining balance that is voluntarily prepaid per year. Refinancing-incentive
shape:

    CPR(r) = clip(cpr_base + beta * max(0, c - r), 0, cpr_cap)

with r the market refi rate at the remaining-term tenor. As market rates fall
below the contract rate, CPR rises and the mortgage effectively shortens.

Monthly mortality:  SMM = 1 - (1 - CPR)^(1/12)

CPR-adjusted schedule (applied each month *after* the scheduled principal):

    B_t = (B_{t-1} - A_t) * (1 - SMM_t)
    prepay_t = (B_{t-1} - A_t) * SMM_t
    total_principal_t = A_t + prepay_t

Conservation: sum(total_principal_t) = N holds exactly (no defaults modelled).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # avoid a circular import via valuation/__init__.py -> eve.py
    from basel_risk_engine.valuation.curve import YieldCurve


class CPRParams(BaseModel):
    """Frozen CPR parameters."""

    model_config = ConfigDict(frozen=True)

    cpr_base: float = Field(
        default=0.06, ge=0.0, le=1.0,
        description="Floor CPR when refi incentive is zero (annualised).",
    )
    beta: float = Field(
        default=8.0, ge=0.0,
        description="Refi sensitivity: CPR adds beta * max(0, c - r).",
    )
    cpr_cap: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Annualised CPR cap; real loans saturate well below 1.",
    )


def level_payment(notional: float, annual_rate: float, term_months: int) -> float:
    """Constant monthly payment that fully amortises `notional` over `term_months`.

    For (sub)denormal rates the closed-form denominator collapses to ~0; we fall
    back to the rate-free principal-only payment, which is the analytic limit.
    """
    if term_months <= 0:
        raise ValueError("term_months must be positive")
    r = annual_rate / 12.0
    if abs(r) < 1e-12:
        return notional / term_months
    return notional * r / (1.0 - (1.0 + r) ** -term_months)


def cpr_curve(
    contract_rate: float,
    refi_rates: np.ndarray,
    params: CPRParams,
) -> np.ndarray:
    """Annualised CPR per month given a refi-rate path (shape (n_months,))."""
    refi = np.asarray(refi_rates, dtype=np.float64)
    incentive = np.maximum(0.0, contract_rate - refi)
    cpr = params.cpr_base + params.beta * incentive
    return np.clip(cpr, 0.0, params.cpr_cap)


def amortisation_schedule(
    notional: float,
    annual_rate: float,
    term_months: int,
    *,
    cpr_per_month: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-month amortisation schedule, optionally with CPR-driven prepayment.

    Columns: month, balance_open, scheduled_payment, interest,
    scheduled_principal, prepay_principal, total_principal, balance_close.

    Truncates at the month the balance is fully repaid (avoids dead rows when
    high CPR shortens the loan early). Conservation holds exactly.
    """
    if cpr_per_month is None:
        cpr_per_month = np.zeros(term_months, dtype=np.float64)
    cpr_arr = np.asarray(cpr_per_month, dtype=np.float64)
    if cpr_arr.shape != (term_months,):
        raise ValueError(f"cpr_per_month shape {cpr_arr.shape} != ({term_months},)")

    smm = 1.0 - (1.0 - cpr_arr) ** (1.0 / 12.0)
    P = level_payment(notional, annual_rate, term_months)
    r_m = annual_rate / 12.0

    months = np.empty(term_months, dtype=np.int64)
    bal_open = np.empty(term_months, dtype=np.float64)
    payment = np.empty(term_months, dtype=np.float64)
    interest = np.empty(term_months, dtype=np.float64)
    sched_principal = np.empty(term_months, dtype=np.float64)
    prepay = np.empty(term_months, dtype=np.float64)
    total_principal = np.empty(term_months, dtype=np.float64)
    bal_close = np.empty(term_months, dtype=np.float64)

    B = notional
    last = term_months
    for t in range(term_months):
        I = B * r_m
        A = min(P - I, B)
        balance_after_sched = B - A
        prepay_t = balance_after_sched * smm[t]
        total_t = A + prepay_t
        B_new = balance_after_sched - prepay_t

        months[t] = t + 1
        bal_open[t] = B
        payment[t] = P
        interest[t] = I
        sched_principal[t] = A
        prepay[t] = prepay_t
        total_principal[t] = total_t
        bal_close[t] = B_new

        B = B_new
        if B < 1e-9:
            last = t + 1
            break

    return pd.DataFrame({
        "month": months[:last],
        "balance_open": bal_open[:last],
        "scheduled_payment": payment[:last],
        "interest": interest[:last],
        "scheduled_principal": sched_principal[:last],
        "prepay_principal": prepay[:last],
        "total_principal": total_principal[:last],
        "balance_close": bal_close[:last],
    })


def project_refi_rates(
    curve: YieldCurve,
    term_months: int,
    *,
    refi_tenor_floor_months: int = 12,
) -> np.ndarray:
    """For each scheduled month t, return the curve's zero yield at the
    remaining contractual tenor as the refi rate.

    Floored at refi_tenor_floor_months to avoid the ultra-short end driving CPR
    in the last months.
    """
    remaining = term_months - np.arange(term_months)
    remaining = np.maximum(remaining, refi_tenor_floor_months)
    return curve.yield_at(remaining / 12.0)


def value_mortgage(
    notional: float,
    contract_rate: float,
    term_months: int,
    curve: YieldCurve,
    *,
    cpr_params: CPRParams | None = None,
) -> dict[str, float]:
    """PV / WAL / CPR-adjusted vs scheduled comparison of one mortgage.

    Discount and refi rates both come from `curve` (no funding-vs-refi spread
    modelled at this stage). The CPR-adjusted PV is the engine's PV; the
    scheduled PV is reported as a no-prepayment counterfactual.
    """
    cpr_params = cpr_params or CPRParams()
    refi = project_refi_rates(curve, term_months)
    cpr = cpr_curve(contract_rate, refi, cpr_params)
    cpr_sched = amortisation_schedule(notional, contract_rate, term_months, cpr_per_month=cpr)
    no_cpr_sched = amortisation_schedule(notional, contract_rate, term_months)

    def _pv(df: pd.DataFrame) -> float:
        taus = df["month"].to_numpy() / 12.0
        cash = df["interest"].to_numpy() + df["total_principal"].to_numpy()
        dfs = curve.discount_factor(taus)
        return float((cash * dfs).sum())

    wal = float(
        ((cpr_sched["month"].to_numpy() / 12.0) * cpr_sched["total_principal"].to_numpy()).sum()
        / cpr_sched["total_principal"].to_numpy().sum()
    )
    return {
        "pv_cpr": _pv(cpr_sched),
        "pv_scheduled": _pv(no_cpr_sched),
        "weighted_avg_life_years": wal,
        "avg_cpr": float(cpr.mean()),
        "effective_term_months": int(len(cpr_sched)),
    }


def value_mortgage_book(
    mortgages: pd.DataFrame,
    curve: YieldCurve,
    *,
    cpr_params: CPRParams | None = None,
) -> pd.DataFrame:
    """Per-mortgage PV table.

    Required columns: cashflow_id, amount, customer_rate, term_months.
    Returns one row per mortgage with PV / WAL / scheduled-PV / avg-CPR.
    """
    required = {"cashflow_id", "amount", "customer_rate", "term_months"}
    missing = required - set(mortgages.columns)
    if missing:
        raise KeyError(f"value_mortgage_book missing columns: {sorted(missing)}")
    cpr_params = cpr_params or CPRParams()

    rows: list[dict] = []
    for r in mortgages.itertuples(index=False):
        d = value_mortgage(
            notional=float(r.amount),
            contract_rate=float(r.customer_rate),
            term_months=int(r.term_months),
            curve=curve,
            cpr_params=cpr_params,
        )
        d["cashflow_id"] = int(r.cashflow_id)
        d["notional"] = float(r.amount)
        d["contract_rate"] = float(r.customer_rate)
        d["term_months"] = int(r.term_months)
        rows.append(d)
    return pd.DataFrame(rows)
