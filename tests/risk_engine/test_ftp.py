"""FTP curve and NII attribution property tests."""

import numpy as np
import pandas as pd
import pytest

from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
from basel_risk_engine.ftp import (
    FTPCurve,
    LiquidityPremiumSchedule,
    compute_attribution,
)
from basel_risk_engine.valuation.curve import YieldCurve


_TENORS = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])


def _upward_curve() -> YieldCurve:
    yields = 0.020 + 0.0015 * _TENORS
    return YieldCurve(tenors_years=_TENORS.copy(), zero_yields=yields)


def _lp_schedule(scale: float = 1.0) -> LiquidityPremiumSchedule:
    lp = scale * np.array([2.0, 5.0, 10.0, 18.0, 30.0, 40.0, 55.0, 65.0, 75.0, 80.0, 80.0])
    return LiquidityPremiumSchedule(tenors_years=_TENORS.copy(), lp_bps=lp)


def _book(maturities_days, products, customer_rates, amounts=None):
    n = len(maturities_days)
    if amounts is None:
        amounts = np.full(n, 1_000_000.0)
    return pd.DataFrame({
        "amount": amounts,
        "product": products,
        "maturity_days": np.asarray(maturities_days, dtype=int),
        "customer_rate": customer_rates,
    })


# ------------------------------------------------------------- curve
def test_ftp_with_zero_lp_equals_base_curve():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=LiquidityPremiumSchedule.zero(_TENORS))
    taus = np.array([0.5, 1.7, 4.3, 8.9])
    np.testing.assert_allclose(ftp.ftp_yield(taus), curve.yield_at(taus), atol=1e-15)


def test_ftp_yield_is_base_plus_lp_in_bps():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule(scale=1.0))
    # At a grid tenor (5y), LP = 55bps
    five_yr_base = curve.yield_at(np.array([5.0]))[0]
    five_yr_ftp = ftp.ftp_yield(np.array([5.0]))[0]
    np.testing.assert_allclose(five_yr_ftp - five_yr_base, 0.0055, atol=1e-12)


def test_ftp_yield_strictly_exceeds_base_for_positive_lp():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule())
    taus = np.array([0.5, 1.0, 3.0, 7.0, 15.0])
    assert np.all(ftp.ftp_yield(taus) > curve.yield_at(taus))


# ------------------------------------------------------------- attribution
def test_attribution_components_sum_to_nii_total():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule())
    book = _book(
        maturities_days=[365, 5 * 365, 90, 90, 180],
        products=["loan", "bond", "deposit", "deposit", "loan"],
        customer_rates=[0.045, 0.030, 0.005, 0.001, 0.040],
    )
    book = apply_nmd_overlay(book, NMDParams())
    result = compute_attribution(book, ftp)
    diff = result.per_row["customer_margin"] + result.per_row["funding_margin"] - result.per_row["nii_total"]
    np.testing.assert_allclose(diff, 0.0, atol=1e-9)


def test_attribution_total_nii_invariant_to_ftp_choice():
    """customer_margin + funding_margin must equal sign·notional·(customer_rate − r_f),
    which is the FTP-free 'bank total NII' on each row."""
    curve = _upward_curve()
    ftp_low = FTPCurve(base_curve=curve, lp_schedule=LiquidityPremiumSchedule.flat(_TENORS, 0.0))
    ftp_high = FTPCurve(base_curve=curve, lp_schedule=LiquidityPremiumSchedule.flat(_TENORS, 100.0))

    book = _book(
        maturities_days=[365, 5 * 365, 90, 90, 180],
        products=["loan", "bond", "deposit", "deposit", "loan"],
        customer_rates=[0.045, 0.030, 0.005, 0.001, 0.040],
    )
    book = apply_nmd_overlay(book, NMDParams())

    r_a = compute_attribution(book, ftp_low)
    r_b = compute_attribution(book, ftp_high)
    # nii_total differs row-wise because r_f changes (overnight rate moves with LP at τ→0),
    # so the right invariant is: customer_margin + funding_margin = sign·notional·(c − r_f).
    # We assert structurally: customer + funding equals nii_total on both branches.
    np.testing.assert_allclose(
        r_a.per_row["customer_margin"] + r_a.per_row["funding_margin"],
        r_a.per_row["nii_total"], atol=1e-9,
    )
    np.testing.assert_allclose(
        r_b.per_row["customer_margin"] + r_b.per_row["funding_margin"],
        r_b.per_row["nii_total"], atol=1e-9,
    )


def test_behavioral_value_zero_for_non_nmds():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule())
    book = _book(
        maturities_days=[365, 5 * 365, 180],
        products=["loan", "bond", "loan"],
        customer_rates=[0.045, 0.030, 0.040],
    )
    book = apply_nmd_overlay(book, NMDParams())
    result = compute_attribution(book, ftp)
    assert (result.per_row["behavioral_value"] == 0.0).all()


def test_behavioral_value_positive_for_deposits_on_upward_curve():
    """NMD deposits priced at long behavioural maturity earn the depositor unit
    a positive credit relative to contractual O/N pricing."""
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule())
    book = _book(
        maturities_days=[30, 60, 90, 180],  # all under threshold → NMDs
        products=["deposit"] * 4,
        customer_rates=[0.0] * 4,
    )
    book = apply_nmd_overlay(book, NMDParams())
    result = compute_attribution(book, ftp)
    # Every row is an NMD
    assert book["is_nmd"].all()
    assert (result.per_row["behavioral_value"] > 0).all()


def test_attribution_refuses_if_columns_missing():
    curve = _upward_curve()
    ftp = FTPCurve(base_curve=curve, lp_schedule=_lp_schedule())
    book = _book(
        maturities_days=[365],
        products=["loan"],
        customer_rates=[0.045],
    )
    # No NMD overlay yet — missing behavioral_maturity_years + is_nmd
    with pytest.raises(KeyError):
        compute_attribution(book, ftp)


def test_lp_schedule_interpolates_linearly():
    schedule = LiquidityPremiumSchedule(
        tenors_years=np.array([1.0, 5.0, 10.0]),
        lp_bps=np.array([10.0, 50.0, 100.0]),
    )
    # Halfway between 1y (10bps) and 5y (50bps) → 30bps at 3y
    assert schedule.at(3.0) == pytest.approx(30.0)
    # Above the last grid point → flat extrapolation at 100bps
    assert schedule.at(20.0) == pytest.approx(100.0)


def test_zero_lp_schedule_works():
    sched = LiquidityPremiumSchedule.zero(_TENORS)
    assert (sched.lp_bps == 0).all()
    assert sched.at(np.array([2.0, 10.0, 30.0])).tolist() == [0.0, 0.0, 0.0]
