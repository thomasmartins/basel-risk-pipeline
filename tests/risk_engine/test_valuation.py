"""Property tests for the EVE engine, yield curve, and NMD overlay."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basel_risk_engine.behavioral.nmd import NMDParams, apply_nmd_overlay
from basel_risk_engine.rate_models import VasicekModel, VasicekParams, simulate_paths
from basel_risk_engine.valuation.curve import EBA_BUCKETS, YieldCurve
from basel_risk_engine.valuation.eve import BCBS368_SHOCKS_BPS, EVEEngine
from basel_risk_engine.valuation.nii import compute_nii_paths


def _make_curve():
    tenors = np.array([0.083, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])
    yields = np.full_like(tenors, 0.025)
    return YieldCurve(tenors_years=tenors, zero_yields=yields)


def _make_book(maturities_yrs, products):
    n = len(maturities_yrs)
    return pd.DataFrame(
        {
            "amount": np.full(n, 1_000_000.0),
            "product": products,
            "maturity_days": (np.asarray(maturities_yrs) * 365).astype(int),
            "behavioral_maturity_years": maturities_yrs,
            "is_nmd": np.zeros(n, dtype=bool),
        }
    )


# ----------------------------------------------------------- yield curve
def test_curve_shift_changes_discount():
    curve = _make_curve()
    shifted = curve.shifted({b: 100.0 for b in EBA_BUCKETS})  # +100bps everywhere
    df0 = curve.discount_factor(np.array([1.0, 5.0, 10.0]))
    df1 = shifted.discount_factor(np.array([1.0, 5.0, 10.0]))
    assert np.all(df1 < df0)


def test_curve_parallel_shift_equals_uniform_bucket_shift():
    curve = _make_curve()
    parallel = curve.parallel_shifted(150.0)
    uniform = curve.shifted({b: 150.0 for b in EBA_BUCKETS})
    np.testing.assert_allclose(parallel.zero_yields, uniform.zero_yields)


# ----------------------------------------------------------- EVE monotonicity
@given(
    n_loans=st.integers(min_value=10, max_value=40),
    n_deposits=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=8, deadline=None)
def test_eve_parallel_up_lowers_eve_for_asset_heavy_book(n_loans, n_deposits):
    """Asset-heavy book + +200bps parallel → ΔEVE < 0."""
    book = pd.concat(
        [
            _make_book([2.0] * n_loans, ["loan"] * n_loans),
            _make_book([0.25] * n_deposits, ["deposit"] * n_deposits),
        ],
        ignore_index=True,
    )
    base = _make_curve()
    engine = EVEEngine(base_curve=base)
    baseline = engine.value(book, base)
    shocked = engine.value(book, base.parallel_shifted(200.0))
    assert shocked < baseline


def test_bcbs368_scenarios_present():
    """BCBS 368 returns exactly the six prescribed shocks."""
    book = _make_book([1.0, 5.0, 10.0], ["loan", "loan", "bond"])
    engine = EVEEngine(base_curve=_make_curve())
    results = engine.bcbs368(book)
    names = {r.scenario for r in results}
    assert names == set(BCBS368_SHOCKS_BPS.keys())
    assert len(results) == 6


def test_supervisory_outlier_test_threshold_logic():
    """A tiny Tier1 forces a breach; a very large Tier1 does not."""
    book = _make_book([5.0, 10.0], ["loan", "bond"])
    engine = EVEEngine(base_curve=_make_curve())
    breach_result = engine.supervisory_outlier_test(book, tier1_capital=1.0)
    safe_result = engine.supervisory_outlier_test(book, tier1_capital=1e15)
    assert breach_result.breach is True
    assert safe_result.breach is False


# ----------------------------------------------------------- NMD overlay
def test_nmd_overlay_stretches_short_deposits():
    book = _make_book([0.25, 0.25, 5.0], ["deposit", "deposit", "loan"])
    out = apply_nmd_overlay(book, NMDParams(stable_core_pct=0.7, core_behavioral_maturity_yrs=5.0))
    # Deposits got stretched: new behavioral > contractual
    deps = out[out["product"] == "deposit"]
    assert (deps["behavioral_maturity_years"] > 0.25).all()
    assert deps["is_nmd"].all()
    # Loans untouched
    loans = out[out["product"] == "loan"]
    assert loans["behavioral_maturity_years"].iloc[0] == pytest.approx(5.0)
    assert not loans["is_nmd"].any()


def test_nmd_overlay_skips_long_deposits():
    book = _make_book([5.0], ["deposit"])
    out = apply_nmd_overlay(book, NMDParams())
    assert not out["is_nmd"].any()
    assert out["behavioral_maturity_years"].iloc[0] == pytest.approx(5.0)


# ----------------------------------------------------------- NII
def test_nii_paths_have_expected_horizons_and_shape():
    model = VasicekModel(VasicekParams(kappa=0.5, theta=0.025, sigma=0.01, r0=0.03))
    paths = simulate_paths(model, n_paths=100, horizon_years=4.0, dt=1 / 12, seed=1)
    book = apply_nmd_overlay(
        _make_book([0.5, 2.0, 5.0], ["deposit", "loan", "bond"]), NMDParams()
    )
    df = compute_nii_paths(book, paths, nmd=NMDParams())
    assert set(df["horizon_months"]) == {12, 24, 36}
    assert (df.groupby("horizon_months").size() == 100).all()
    # repricing_gap is a scalar per horizon
    assert df.groupby("horizon_months")["repricing_gap"].nunique().eq(1).all()
