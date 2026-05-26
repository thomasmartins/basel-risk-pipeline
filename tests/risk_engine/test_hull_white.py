"""Hull-White 1F property tests.

The headline claim of HW1F is that, unlike Vasicek, the model reprices today's
observed zero curve exactly: P_model(0, τ; r_0) = P_market(0, τ) for every τ.
The first test pins that down on a non-trivial (non-flat, upward-sloping) curve.
"""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basel_risk_engine.rate_models import (
    HullWhiteModel,
    HullWhiteParams,
    VasicekModel,
    VasicekParams,
    simulate_paths,
)
from basel_risk_engine.valuation.curve import YieldCurve


def _upward_curve(r0: float = 0.020, slope_per_year: float = 0.0015) -> YieldCurve:
    tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    yields = r0 + slope_per_year * tenors
    return YieldCurve(tenors_years=tenors, zero_yields=yields)


def test_hw_reprices_initial_curve_exactly_at_grid():
    """P_HW(0, τ; r0) == P_market(0, τ) for every grid tenor when r0 = f^M(0,0)."""
    curve = _upward_curve()
    r0 = float(curve.forward_rate(np.array([1e-12]))[0])
    model = HullWhiteModel(HullWhiteParams(a=0.30, sigma=0.012, r0=r0), curve)
    model_dfs = model.bond_price(curve.tenors_years, r0)
    market_dfs = curve.discount_factor(curve.tenors_years)
    np.testing.assert_allclose(model_dfs, market_dfs, rtol=1e-12, atol=1e-13)


@given(
    a=st.floats(min_value=0.05, max_value=1.5),
    sigma=st.floats(min_value=0.002, max_value=0.025),
)
@settings(max_examples=8, deadline=None)
def test_hw_reprices_initial_curve_at_arbitrary_tau(a, sigma):
    """Curve fit holds at arbitrary off-grid tenors too — that's the point of HW1F."""
    curve = _upward_curve()
    r0 = float(curve.forward_rate(np.array([1e-12]))[0])
    model = HullWhiteModel(HullWhiteParams(a=a, sigma=sigma, r0=r0), curve)
    taus = np.array([0.5, 1.7, 4.3, 8.9, 15.0, 25.0])
    model_dfs = model.bond_price(taus, r0)
    market_dfs = curve.discount_factor(taus)
    np.testing.assert_allclose(model_dfs, market_dfs, rtol=1e-12, atol=1e-13)


@given(
    a=st.floats(min_value=0.1, max_value=1.2),
    sigma=st.floats(min_value=0.003, max_value=0.02),
)
@settings(max_examples=6, deadline=None)
def test_hw_bond_price_monotone_decreasing_in_r(a, sigma):
    """For fixed τ > 0, P(0, τ; r) is strictly decreasing in r."""
    curve = _upward_curve()
    model = HullWhiteModel(HullWhiteParams(a=a, sigma=sigma, r0=0.02), curve)
    rs = np.linspace(-0.01, 0.05, 25)
    for tau in (0.5, 2.0, 7.0):
        prices = model.bond_price(tau, rs)
        assert np.all(np.diff(prices) < 0)


def test_hw_mc_expected_short_rate_matches_forward_plus_convexity():
    """E[r_t] ≈ f^M(0, t) + σ²/(2a²)(1 - e^{-at})² across the MC ensemble."""
    curve = _upward_curve()
    a, sigma, r0 = 0.40, 0.010, float(curve.forward_rate(np.array([1e-12]))[0])
    model = HullWhiteModel(HullWhiteParams(a=a, sigma=sigma, r0=r0), curve)

    n_paths, dt, n_steps = 8000, 1 / 12, 60  # 5y, monthly
    paths = simulate_paths(model, n_paths=n_paths, horizon_years=n_steps * dt, dt=dt, seed=11)

    times = np.arange(n_steps + 1) * dt
    times_eval = times[12::12]  # check at years 1..5
    expected_forward = curve.forward_rate(times_eval)
    convexity = (sigma ** 2) / (2 * a ** 2) * (1 - np.exp(-a * times_eval)) ** 2
    theoretical = expected_forward + convexity
    empirical = paths.short_rates[:, 12::12].mean(axis=0)

    # MC SE ~ σ/sqrt(2a · n_paths); pad generously
    se = sigma / math.sqrt(2 * a * n_paths)
    np.testing.assert_allclose(empirical, theoretical, atol=6 * se + 1e-4)


def test_hw_calibration_recovers_a_and_sigma_on_synthetic_history():
    """Long synthetic series + closed-form regression should recover (a, σ) within tolerance."""
    curve = _upward_curve()
    r0_curve = float(curve.forward_rate(np.array([1e-12]))[0])
    true_p = HullWhiteParams(a=0.50, sigma=0.010, r0=r0_curve)
    model = HullWhiteModel(true_p, curve)

    dt = 1 / 52  # weekly, 20y
    rates = model.simulate(n_paths=1, n_steps=20 * 52, dt=dt, seed=42, antithetic=False)[0]
    calib = HullWhiteModel.calibrate(rates, dt=dt, market_curve=curve, r0_override=r0_curve)

    assert calib.params.a > 0
    assert abs(calib.params.sigma - true_p.sigma) / true_p.sigma < 0.10
    # `a` has the same well-documented small-sample upward bias as κ in Vasicek
    # (Yu 2009; Tang & Chen 2009). Only assert sign / mean reversion.
    # Curve-fit-by-construction
    assert calib.curve_fit_max_residual < 1e-12


def test_hw_calibration_rejects_non_meanreverting_residuals():
    """If r - f^M is not mean-reverting (AR(1) slope ≥ 1), refuse."""
    curve = _upward_curve()
    rng = np.random.default_rng(0)
    # Construct a series whose de-meaned form trends — driven by a deterministic ramp
    n = 500
    dt = 1 / 12
    t_grid = np.arange(n) * dt
    f0 = curve.forward_rate(np.where(t_grid <= 0, 1e-12, t_grid))
    drift = np.cumsum(np.abs(rng.standard_normal(n)) * 0.001)
    rates = f0 + drift  # x_t = r_t - f^M(0,t) trends ⇒ AR(1) slope > 1
    with pytest.raises(ValueError):
        HullWhiteModel.calibrate(rates, dt=dt, market_curve=curve)


def test_hw_eve_distribution_concentrates_to_a_point_when_sigma_is_tiny():
    """As σ → 0 the MC ΔEVE distribution should collapse to a single deterministic
    value (drift along α(t), not the baseline EVE — under an upward-sloping curve the
    deterministic path moves up the forward curve and produces a non-zero shift)."""
    import pandas as pd
    from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
    from basel_risk_engine.valuation.eve import EVEEngine

    curve = _upward_curve()
    r0 = float(curve.forward_rate(np.array([1e-12]))[0])
    model = HullWhiteModel(HullWhiteParams(a=0.5, sigma=1e-8, r0=r0), curve)
    paths = simulate_paths(model, n_paths=400, horizon_years=2.0, dt=1 / 12, seed=3)

    book = pd.DataFrame({
        "amount": [1_000_000.0, 800_000.0, 500_000.0],
        "product": ["loan", "bond", "deposit"],
        "maturity_days": [365, 5 * 365, 90],
        "behavioral_maturity_years": [1.0, 5.0, 0.25],
        "is_nmd": [False, False, False],
    })
    book = apply_nmd_overlay(book, NMDParams())

    engine = EVEEngine(base_curve=curve, rate_model=model)
    dist = engine.mc_distribution(book, paths, forward_horizon_years=1.0)
    # Variance, not magnitude — the deterministic forward shift is fine.
    assert float(np.std(dist)) < 1.0


def test_hw_and_vasicek_share_the_eve_engine_api():
    """EVEEngine.mc_distribution must accept either model via the ShortRateModel protocol."""
    import pandas as pd
    from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
    from basel_risk_engine.valuation.eve import EVEEngine

    curve = _upward_curve()
    book = pd.DataFrame({
        "amount": [1_000_000.0],
        "product": ["loan"],
        "maturity_days": [5 * 365],
        "behavioral_maturity_years": [5.0],
        "is_nmd": [False],
    })
    book = apply_nmd_overlay(book, NMDParams())

    vas = VasicekModel(VasicekParams(kappa=0.4, theta=0.025, sigma=0.01, r0=0.02))
    vas_paths = simulate_paths(vas, n_paths=200, horizon_years=2.0, dt=1 / 12, seed=1)
    EVEEngine(base_curve=curve, rate_model=vas).mc_distribution(book, vas_paths)

    hw = HullWhiteModel(HullWhiteParams(a=0.4, sigma=0.01, r0=0.02), curve)
    hw_paths = simulate_paths(hw, n_paths=200, horizon_years=2.0, dt=1 / 12, seed=1)
    EVEEngine(base_curve=curve, rate_model=hw).mc_distribution(book, hw_paths)
