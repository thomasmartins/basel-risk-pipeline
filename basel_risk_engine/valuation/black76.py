"""Black-76 / Brigo-Mercurio closed-form pricing of European bond options
under Hull-White 1F.

For a European call with expiry T and strike K (in unit-notional terms) on a
zero-coupon bond maturing at S > T, the HW1F closed form (Brigo & Mercurio,
*Interest Rate Models*, 2nd ed., §3.3) is

    ZBC(0, T, S, K) = P(0,S) * N(h) - K * P(0,T) * N(h - sigma_P)

with
    B(T, S)  = (1 - exp(-a (S - T))) / a
    sigma_P  = sigma * B(T, S) * sqrt((1 - exp(-2 a T)) / (2 a))
    h        = (1 / sigma_P) * ln(P(0,S) / (K * P(0,T))) + sigma_P / 2

sigma_P is the *integrated* lognormal volatility of the forward bond price
over [0, T]; the formula collapses to the no-vol intrinsic
    max(P(0,S) - K * P(0,T), 0)
when sigma -> 0 or T -> 0.

Callable-bond convention here:
    A bond with notional N maturing at T_S pays N at T_S (zero-coupon stylised);
    the issuer's call at T_call <= T_S with strike call_strike_pct of par
    (call_strike_pct = 100.0 means par) is worth
        call_value = N * ZBC(0, T_call, T_S, call_strike_pct / 100.0)
    and the holder's callable PV is
        callable_pv = straight_pv - call_value
                    = N * P(0, T_S) - call_value.

The unit-notional convention isolates the option value from the notional and
matches the synthetic schema: bonds in `cashflows` carry a bullet notional
`amount` at maturity, and `call_strike_pct` is quoted in % of par.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:  # break the rate_models <-> valuation cycle
    from basel_risk_engine.rate_models.hull_white import HullWhiteModel
    from basel_risk_engine.valuation.curve import YieldCurve


def hw_bond_option_integrated_vol(a: float, sigma: float, T: float, S: float) -> float:
    """sigma_P over [0, T] for a zero maturing at S, under HW1F params (a, sigma)."""
    if T <= 0.0 or S <= T:
        return 0.0
    B_TS = (1.0 - math.exp(-a * (S - T))) / a
    integ = (1.0 - math.exp(-2.0 * a * T)) / (2.0 * a)
    return sigma * B_TS * math.sqrt(integ)


def zbc_price(
    curve: YieldCurve,
    a: float,
    sigma: float,
    T_call: float,
    T_mat: float,
    strike_unit: float,
) -> float:
    """Price of a European call on a unit zero-coupon bond P(.,T_mat) struck at
    `strike_unit` at expiry T_call. Returns 0 for degenerate inputs.
    """
    if T_call <= 0.0 or T_mat <= T_call or strike_unit <= 0.0:
        return 0.0
    P0_T = float(curve.discount_factor(np.array([T_call]))[0])
    P0_S = float(curve.discount_factor(np.array([T_mat]))[0])
    sigma_P = hw_bond_option_integrated_vol(a, sigma, T_call, T_mat)
    if sigma_P <= 1e-12:
        return max(P0_S - strike_unit * P0_T, 0.0)
    h = math.log(P0_S / (strike_unit * P0_T)) / sigma_P + sigma_P / 2.0
    return P0_S * norm.cdf(h) - strike_unit * P0_T * norm.cdf(h - sigma_P)


@dataclass(frozen=True)
class CallableBondResult:
    cashflow_id: int
    notional: float
    straight_pv: float
    call_value: float
    callable_pv: float
    t_call_years: float
    t_mat_years: float
    strike_unit: float
    integrated_vol: float


def value_callable_bond(
    *,
    cashflow_id: int,
    notional: float,
    t_call_years: float,
    t_mat_years: float,
    call_strike_pct: float,
    curve: YieldCurve,
    hw_model: HullWhiteModel,
) -> CallableBondResult:
    strike_unit = call_strike_pct / 100.0
    a = hw_model.params.a
    sigma = hw_model.params.sigma
    straight_pv = notional * float(curve.discount_factor(np.array([t_mat_years]))[0])
    call_value = notional * zbc_price(curve, a, sigma, t_call_years, t_mat_years, strike_unit)
    return CallableBondResult(
        cashflow_id=cashflow_id,
        notional=notional,
        straight_pv=straight_pv,
        call_value=call_value,
        callable_pv=straight_pv - call_value,
        t_call_years=t_call_years,
        t_mat_years=t_mat_years,
        strike_unit=strike_unit,
        integrated_vol=hw_bond_option_integrated_vol(a, sigma, t_call_years, t_mat_years),
    )


def value_callable_book(
    callable_bonds: "pd.DataFrame",
    curve: YieldCurve,
    hw_model: HullWhiteModel,
) -> "pd.DataFrame":
    """Per-bond PV / call value / callable PV.

    Required columns: cashflow_id, amount (notional), t_call_years,
    t_mat_years, call_strike_pct.
    """
    import pandas as pd

    required = {"cashflow_id", "amount", "t_call_years", "t_mat_years", "call_strike_pct"}
    missing = required - set(callable_bonds.columns)
    if missing:
        raise KeyError(f"value_callable_book missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for r in callable_bonds.itertuples(index=False):
        res = value_callable_bond(
            cashflow_id=int(r.cashflow_id),
            notional=float(r.amount),
            t_call_years=float(r.t_call_years),
            t_mat_years=float(r.t_mat_years),
            call_strike_pct=float(r.call_strike_pct),
            curve=curve,
            hw_model=hw_model,
        )
        rows.append({
            "cashflow_id": res.cashflow_id,
            "notional": res.notional,
            "t_call_years": res.t_call_years,
            "t_mat_years": res.t_mat_years,
            "strike_unit": res.strike_unit,
            "integrated_vol": res.integrated_vol,
            "straight_pv": res.straight_pv,
            "call_value": res.call_value,
            "callable_pv": res.callable_pv,
        })
    return pd.DataFrame(rows)
