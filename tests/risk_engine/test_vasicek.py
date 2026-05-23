"""Vasicek model behavioural / property-based tests."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basel_risk_engine.rate_models import VasicekModel, VasicekParams, simulate_paths


@given(
    kappa=st.floats(min_value=0.05, max_value=2.0),
    theta=st.floats(min_value=-0.02, max_value=0.08),
    sigma=st.floats(min_value=0.001, max_value=0.03),
    r0=st.floats(min_value=-0.02, max_value=0.10),
)
@settings(max_examples=15, deadline=None)
def test_mc_terminal_mean_converges_to_theta(kappa, theta, sigma, r0):
    """Under risk-neutral Vasicek with large enough horizon, E[r_T] → θ."""
    p = VasicekParams(kappa=kappa, theta=theta, sigma=sigma, r0=r0)
    paths = simulate_paths(
        VasicekModel(p),
        n_paths=4000,
        horizon_years=max(20.0, 6 / kappa),  # well past half-life
        dt=1 / 12,
        seed=1,
    )
    terminal_mean = float(paths.short_rates[:, -1].mean())
    # MC SE ~ sigma / sqrt(2 kappa * n_paths). Add generous slack.
    se = sigma / math.sqrt(2 * kappa * 4000)
    assert abs(terminal_mean - theta) < 6 * se + 1e-4


@given(
    kappa=st.floats(min_value=0.1, max_value=1.5),
    theta=st.floats(min_value=0.005, max_value=0.05),
    sigma=st.floats(min_value=0.003, max_value=0.02),
)
@settings(max_examples=10, deadline=None)
def test_bond_price_monotone_in_tau(kappa, theta, sigma):
    """For non-negative implied yields, P(0,T) is monotone decreasing in T."""
    model = VasicekModel(VasicekParams(kappa=kappa, theta=theta, sigma=sigma, r0=theta))
    taus = np.linspace(0.0, 30.0, 31)
    bonds = model.bond_price(taus, theta)
    # Strict monotonicity past the trivial tau=0 case
    assert np.all(np.diff(bonds) <= 1e-12)


def test_calibration_roundtrip_recovers_theta_sigma():
    """Long synthetic series + closed-form OLS calibration recovers θ and σ within tolerance.

    κ is well-known to have meaningful small-sample bias (Yu 2009), so we
    only assert its sign / order-of-magnitude.
    """
    true_p = VasicekParams(kappa=0.5, theta=0.025, sigma=0.01, r0=0.03)
    model = VasicekModel(true_p)
    dt = 1 / 52  # weekly, 20 years
    rates = model.simulate(n_paths=1, n_steps=20 * 52, dt=dt, seed=42, antithetic=False)[0]
    calib = VasicekModel.calibrate(rates, dt=dt)
    assert calib.params.kappa > 0
    assert abs(calib.params.theta - true_p.theta) < 0.005
    assert abs(calib.params.sigma - true_p.sigma) / true_p.sigma < 0.10


def test_calibration_rejects_non_meanreverting():
    """If the regressor coefficient is ≥ 1 the OLS calibration must refuse."""
    # Strictly trending series — AR(1) slope > 1.
    rng = np.random.default_rng(0)
    drift = np.cumsum(np.abs(rng.standard_normal(500)) * 0.001)
    rates = 0.01 + drift
    with pytest.raises(ValueError):
        VasicekModel.calibrate(rates, dt=1 / 12)
