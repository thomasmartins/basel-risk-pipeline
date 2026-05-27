"""Property tests for Black-76 callable bond pricing under HW1F."""

import numpy as np
import pytest

from basel_risk_engine.rate_models.hull_white import HullWhiteModel, HullWhiteParams
from basel_risk_engine.valuation.black76 import (
    hw_bond_option_integrated_vol,
    value_callable_bond,
    zbc_price,
)
from basel_risk_engine.valuation.curve import YieldCurve


_TENORS = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def _flat_curve(y: float = 0.03) -> YieldCurve:
    return YieldCurve(tenors_years=_TENORS.copy(), zero_yields=np.full_like(_TENORS, y))


def _hw_model(curve: YieldCurve, *, a: float = 0.1, sigma: float = 0.01) -> HullWhiteModel:
    return HullWhiteModel(HullWhiteParams(a=a, sigma=sigma, r0=0.03), curve)


# ----------------------------------------------------------- integrated vol
def test_integrated_vol_zero_when_sigma_is_zero():
    # sigma=0 not allowed by HullWhiteParams (gt=0), so probe the helper directly
    assert hw_bond_option_integrated_vol(a=0.1, sigma=0.0, T=2.0, S=5.0) == 0.0


def test_integrated_vol_zero_when_no_time_to_expiry():
    assert hw_bond_option_integrated_vol(a=0.1, sigma=0.01, T=0.0, S=5.0) == 0.0


def test_integrated_vol_zero_when_call_at_or_after_maturity():
    assert hw_bond_option_integrated_vol(a=0.1, sigma=0.01, T=5.0, S=5.0) == 0.0
    assert hw_bond_option_integrated_vol(a=0.1, sigma=0.01, T=5.0, S=4.0) == 0.0


def test_integrated_vol_monotone_in_sigma():
    iv_low = hw_bond_option_integrated_vol(a=0.1, sigma=0.005, T=2.0, S=5.0)
    iv_high = hw_bond_option_integrated_vol(a=0.1, sigma=0.020, T=2.0, S=5.0)
    assert iv_high > iv_low


# ----------------------------------------------------------- zbc edge cases
def test_zbc_returns_zero_for_degenerate_inputs():
    curve = _flat_curve()
    assert zbc_price(curve, a=0.1, sigma=0.01, T_call=0.0, T_mat=5.0, strike_unit=1.0) == 0.0
    assert zbc_price(curve, a=0.1, sigma=0.01, T_call=5.0, T_mat=5.0, strike_unit=1.0) == 0.0
    assert zbc_price(curve, a=0.1, sigma=0.01, T_call=6.0, T_mat=5.0, strike_unit=1.0) == 0.0


def test_zbc_collapses_to_intrinsic_when_vol_is_tiny():
    curve = _flat_curve(y=0.03)
    P0_T = float(curve.discount_factor(np.array([2.0]))[0])
    P0_S = float(curve.discount_factor(np.array([5.0]))[0])
    strike = 1.0  # par
    intrinsic = max(P0_S - strike * P0_T, 0.0)
    # Use a sigma small enough to be essentially zero against this formula
    c = zbc_price(curve, a=0.1, sigma=1e-10, T_call=2.0, T_mat=5.0, strike_unit=strike)
    assert c == pytest.approx(intrinsic, abs=1e-9)


def test_zbc_monotone_in_sigma():
    curve = _flat_curve(y=0.03)
    c_low = zbc_price(curve, a=0.1, sigma=0.005, T_call=2.0, T_mat=5.0, strike_unit=1.0)
    c_high = zbc_price(curve, a=0.1, sigma=0.020, T_call=2.0, T_mat=5.0, strike_unit=1.0)
    assert c_high > c_low


def test_zbc_decreases_in_strike():
    curve = _flat_curve(y=0.03)
    c_low_strike = zbc_price(curve, a=0.1, sigma=0.01, T_call=2.0, T_mat=5.0, strike_unit=0.80)
    c_high_strike = zbc_price(curve, a=0.1, sigma=0.01, T_call=2.0, T_mat=5.0, strike_unit=1.20)
    assert c_low_strike > c_high_strike


def test_zbc_bounded_by_underlying_bond_value():
    """A European call on P(.,S) is bounded above by P(0,S) (no-arbitrage)."""
    curve = _flat_curve(y=0.03)
    P0_S = float(curve.discount_factor(np.array([5.0]))[0])
    c = zbc_price(curve, a=0.1, sigma=0.10, T_call=2.0, T_mat=5.0, strike_unit=0.01)
    assert c <= P0_S + 1e-12


# ----------------------------------------------------------- callable bond
def test_callable_pv_below_straight_pv_when_call_has_value():
    curve = _flat_curve(y=0.03)
    hw = _hw_model(curve, a=0.1, sigma=0.02)
    res = value_callable_bond(
        cashflow_id=1,
        notional=100_000.0,
        t_call_years=2.0,
        t_mat_years=5.0,
        call_strike_pct=100.0,
        curve=curve,
        hw_model=hw,
    )
    assert res.call_value > 0
    assert res.callable_pv < res.straight_pv
    assert res.callable_pv == pytest.approx(res.straight_pv - res.call_value)


def test_callable_pv_converges_to_straight_pv_when_vol_is_tiny():
    curve = _flat_curve(y=0.03)
    # Very low vol -> call has near-zero time value beyond intrinsic.
    hw = _hw_model(curve, a=0.1, sigma=0.0001)
    res = value_callable_bond(
        cashflow_id=1,
        notional=100_000.0,
        t_call_years=2.0,
        t_mat_years=5.0,
        call_strike_pct=100.0,
        curve=curve,
        hw_model=hw,
    )
    # On a flat curve at 3% with a par strike at T_call=2y on a 5y bond, intrinsic =
    # P(0,5) - 1 * P(0,2). With flat 3% yields, P(0,2) > P(0,5), so intrinsic = 0
    # and the deep-out-of-the-money call is worth essentially nothing.
    assert res.call_value < 1e-6 * res.notional
    assert res.callable_pv == pytest.approx(res.straight_pv, rel=1e-6)


def test_callable_pv_converges_to_straight_pv_when_strike_is_extreme():
    """Deep-out-of-the-money strike (call price >> bond price) -> call worthless."""
    curve = _flat_curve(y=0.03)
    hw = _hw_model(curve, a=0.1, sigma=0.02)
    res = value_callable_bond(
        cashflow_id=1,
        notional=100_000.0,
        t_call_years=2.0,
        t_mat_years=5.0,
        call_strike_pct=300.0,  # 3x par — call will never be in the money
        curve=curve,
        hw_model=hw,
    )
    assert res.call_value < 1e-6 * res.notional
    assert res.callable_pv == pytest.approx(res.straight_pv, rel=1e-6)


def test_callable_bond_scales_linearly_with_notional():
    curve = _flat_curve(y=0.03)
    hw = _hw_model(curve, a=0.1, sigma=0.02)
    common = dict(t_call_years=2.0, t_mat_years=5.0, call_strike_pct=100.0,
                  curve=curve, hw_model=hw)
    r1 = value_callable_bond(cashflow_id=1, notional=100_000.0, **common)
    r2 = value_callable_bond(cashflow_id=1, notional=300_000.0, **common)
    assert r2.straight_pv == pytest.approx(3 * r1.straight_pv)
    assert r2.call_value == pytest.approx(3 * r1.call_value)
    assert r2.callable_pv == pytest.approx(3 * r1.callable_pv)


def test_callable_bond_call_value_is_positive():
    """For any reasonable HW1F parameters and a near-par strike, the
    European call has strictly positive value (time value > 0)."""
    curve = _flat_curve(y=0.03)
    hw = _hw_model(curve, a=0.1, sigma=0.02)
    res = value_callable_bond(
        cashflow_id=1,
        notional=100_000.0,
        t_call_years=2.0,
        t_mat_years=5.0,
        call_strike_pct=100.0,
        curve=curve,
        hw_model=hw,
    )
    assert res.call_value > 0
