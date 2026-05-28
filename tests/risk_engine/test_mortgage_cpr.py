"""Property tests for the mortgage CPR + amortisation module."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basel_risk_engine.behavioral.mortgage_cpr import (
    CPRParams,
    amortisation_schedule,
    cpr_curve,
    level_payment,
    project_refi_rates,
    value_mortgage,
)
from basel_risk_engine.valuation.curve import YieldCurve


_TENORS = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def _upward_curve(level: float = 0.020, slope: float = 0.0015) -> YieldCurve:
    yields = level + slope * _TENORS
    return YieldCurve(tenors_years=_TENORS.copy(), zero_yields=yields)


def _flat_curve(y: float = 0.04) -> YieldCurve:
    return YieldCurve(tenors_years=_TENORS.copy(), zero_yields=np.full_like(_TENORS, y))


# ----------------------------------------------------------- level payment
def test_level_payment_zero_rate_is_uniform():
    """Zero contract rate -> P = N / term, by construction."""
    P = level_payment(120_000.0, 0.0, 360)
    assert P == pytest.approx(120_000.0 / 360)


def test_level_payment_positive_rate_above_principal_only():
    P_rate = level_payment(120_000.0, 0.04, 360)
    P_zero = level_payment(120_000.0, 0.0, 360)
    assert P_rate > P_zero


def test_level_payment_rejects_nonpositive_term():
    with pytest.raises(ValueError):
        level_payment(100_000.0, 0.04, 0)


# ----------------------------------------------------------- schedule conservation
@given(
    notional=st.floats(min_value=10_000.0, max_value=1_000_000.0),
    rate=st.floats(min_value=0.0, max_value=0.12),
    term=st.integers(min_value=12, max_value=360),
)
@settings(max_examples=20, deadline=None)
def test_scheduled_principal_sums_to_notional(notional, rate, term):
    """No-CPR schedule: total principal sums to the notional within fp drift.

    Sub-1bp rates trigger the zero-rate fallback (P = N/term) which accumulates
    Σ A_t = term · (N/term) over many monthly iterations — that loses a few
    ulps for non-power-of-two notionals, hence the small absolute tolerance.
    """
    sched = amortisation_schedule(notional, rate, term)
    assert sched["total_principal"].sum() == pytest.approx(notional, rel=1e-6, abs=0.5)
    assert sched["balance_close"].iloc[-1] == pytest.approx(0.0, abs=0.5)


@given(
    notional=st.floats(min_value=10_000.0, max_value=1_000_000.0),
    rate=st.floats(min_value=0.0, max_value=0.12),
    term=st.integers(min_value=12, max_value=360),
    cpr_level=st.floats(min_value=0.0, max_value=0.50),
)
@settings(max_examples=20, deadline=None)
def test_cpr_schedule_conserves_notional(notional, rate, term, cpr_level):
    """CPR-adjusted schedule still sums to the notional (no defaults modelled)."""
    cpr = np.full(term, cpr_level)
    sched = amortisation_schedule(notional, rate, term, cpr_per_month=cpr)
    assert sched["total_principal"].sum() == pytest.approx(notional, rel=1e-9, abs=1e-3)


def test_cpr_schedule_truncates_when_balance_is_paid_off():
    """With heavy CPR the loan amortises in far fewer months than the term."""
    sched = amortisation_schedule(
        100_000.0, 0.04, 360, cpr_per_month=np.full(360, 0.5),
    )
    assert len(sched) < 360
    assert sched["balance_close"].iloc[-1] == pytest.approx(0.0, abs=1e-3)


# ----------------------------------------------------------- CPR curve
def test_cpr_curve_floors_at_base_when_no_incentive():
    params = CPRParams(cpr_base=0.06, beta=8.0, cpr_cap=0.6)
    # refi >= contract -> incentive = 0 -> CPR = base
    refi = np.array([0.05, 0.06, 0.10])
    cpr = cpr_curve(0.04, refi, params)
    np.testing.assert_allclose(cpr, 0.06)


def test_cpr_curve_increases_as_rates_fall():
    """In-the-money refi: CPR strictly exceeds the base level."""
    params = CPRParams(cpr_base=0.06, beta=10.0, cpr_cap=0.99)
    cpr_no_incentive = cpr_curve(0.04, np.array([0.05]), params)[0]
    cpr_deep_incentive = cpr_curve(0.04, np.array([0.005]), params)[0]
    assert cpr_deep_incentive > cpr_no_incentive
    # 0.06 + 10 * (0.04 - 0.005) = 0.06 + 0.35 = 0.41
    assert cpr_deep_incentive == pytest.approx(0.41)


def test_cpr_curve_respects_cap():
    params = CPRParams(cpr_base=0.10, beta=100.0, cpr_cap=0.30)
    # Massive incentive would push CPR way above 1; cap keeps it at 0.30.
    cpr = cpr_curve(0.10, np.array([0.0]), params)[0]
    assert cpr == pytest.approx(0.30)


# ----------------------------------------------------------- WAL shortening
def test_higher_cpr_shortens_weighted_average_life():
    """Same loan, two scenarios: higher CPR -> shorter WAL."""
    notional, rate, term = 200_000.0, 0.04, 240
    low = amortisation_schedule(notional, rate, term, cpr_per_month=np.full(term, 0.03))
    high = amortisation_schedule(notional, rate, term, cpr_per_month=np.full(term, 0.30))

    def _wal(df):
        return ((df["month"] / 12.0) * df["total_principal"]).sum() / df["total_principal"].sum()

    assert _wal(high) < _wal(low)


# ----------------------------------------------------------- value_mortgage
def test_value_mortgage_on_flat_curve_recovers_par():
    """On a flat curve at the contract rate, a level-payment fixed-rate mortgage
    is worth par (notional) regardless of prepayment timing — the discount
    factor exactly offsets the interest accumulated at that same rate."""
    rate = 0.04
    curve = _flat_curve(rate)
    res = value_mortgage(100_000.0, rate, 120, curve)  # 10-year mortgage at 4%
    assert res["pv_scheduled"] == pytest.approx(100_000.0, rel=1e-3)
    # CPR-adjusted PV also equals par on the flat-curve / no-incentive case:
    # refi rate equals contract rate so incentive = 0; CPR is constant at base
    # and prepayments are also discounted at the contract rate.
    assert res["pv_cpr"] == pytest.approx(100_000.0, rel=1e-3)


def test_value_mortgage_below_par_when_market_yield_above_contract():
    """If market refi yields > contract rate, the bondholder is stuck with a
    below-market coupon, so PV < notional."""
    contract = 0.030
    market = 0.060
    curve = _flat_curve(market)
    res = value_mortgage(100_000.0, contract, 240, curve)
    assert res["pv_cpr"] < 100_000.0
    assert res["pv_scheduled"] < 100_000.0


def test_value_mortgage_returns_finite_values():
    curve = _upward_curve()
    res = value_mortgage(250_000.0, 0.045, 360, curve)
    for k, v in res.items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v), f"non-finite {k}: {v}"


# ----------------------------------------------------------- refi-rate projection
def test_project_refi_rates_uses_curve_at_remaining_tenors():
    curve = _upward_curve(level=0.02, slope=0.001)
    refi = project_refi_rates(curve, term_months=120)
    # First month: 120 months remaining = 10y -> y = 0.02 + 0.001 * 10 = 0.030
    # Last month: 1 month remaining, floored to 12 months -> y at 1y = 0.021
    assert refi[0] == pytest.approx(0.02 + 0.001 * 10, rel=1e-9)
    assert refi[-1] == pytest.approx(0.02 + 0.001 * 1.0, rel=1e-9)
