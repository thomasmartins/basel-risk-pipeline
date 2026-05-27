"""Property tests for the ALMM-style liquidity survival horizon."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from basel_risk_engine.liquidity import (
    LIQUIDITY_STRESS_SCENARIOS,
    LiquidityStressParams,
    compute_survival_horizon,
)


def _make_book(
    outflow_days: list[int],
    outflow_amounts: list[float],
    inflow_days: list[int] | None = None,
    inflow_amounts: list[float] | None = None,
    hqla_stock_amount: float = 0.0,
    hqla_type: str = "Level1",
    deposit_counterparty: str = "retail",
    is_nmd_flag: bool = False,
) -> pd.DataFrame:
    rows = []
    for d, a in zip(outflow_days, outflow_amounts):
        rows.append({
            "maturity_days": int(d), "amount": float(a),
            "direction": "outflow", "product": "deposit",
            "counterparty": deposit_counterparty,
            "hqla_type": "None", "is_nmd": bool(is_nmd_flag),
        })
    for d, a in zip(inflow_days or [], inflow_amounts or []):
        rows.append({
            "maturity_days": int(d), "amount": float(a),
            "direction": "inflow", "product": "loan",
            "counterparty": "retail",
            "hqla_type": "None", "is_nmd": False,
        })
    if hqla_stock_amount > 0:
        rows.append({
            "maturity_days": 365, "amount": float(hqla_stock_amount),
            "direction": "inflow", "product": "bond",
            "counterparty": "wholesale",
            "hqla_type": hqla_type, "is_nmd": False,
        })
    return pd.DataFrame(rows)


def _stress(**overrides) -> LiquidityStressParams:
    base = LIQUIDITY_STRESS_SCENARIOS["idiosyncratic"].model_dump()
    base.update(overrides)
    return LiquidityStressParams(**base)


# --------------------------------------------------------- happy path
def test_no_outflows_yields_horizon_at_max():
    """No outflows + any HQLA stock -> never breaches; horizon = max."""
    book = _make_book(outflow_days=[], outflow_amounts=[], hqla_stock_amount=1_000_000)
    result = compute_survival_horizon(book, stress=_stress(), max_horizon_days=180)
    assert not result.is_breached
    assert result.survival_horizon_days == 180
    assert result.peak_deficit == 0.0


def test_zero_cbc_and_outflow_day_one_breaches_immediately():
    """No HQLA + an outflow on day 1 -> breach at day 1."""
    book = _make_book(
        outflow_days=[1], outflow_amounts=[100_000],
        hqla_stock_amount=0.0,
        deposit_counterparty="wholesale",  # 40% wholesale runoff
    )
    result = compute_survival_horizon(book, stress=_stress(), max_horizon_days=30)
    assert result.is_breached
    assert result.survival_horizon_days == 1
    assert result.peak_deficit < 0


def test_cbc_just_covers_outflow_no_breach():
    """CBC exactly equals the stressed outflow -> running_cbc bottoms at 0,
    not negative, so no breach."""
    outflow = 100_000.0
    # Wholesale 40% runoff => stressed outflow = 40_000. Level1 haircut = 0% in
    # the idiosyncratic preset, so we need >= 40_000 of Level1 stock.
    book = _make_book(
        outflow_days=[5], outflow_amounts=[outflow],
        hqla_stock_amount=40_000.0,
        deposit_counterparty="wholesale",
    )
    result = compute_survival_horizon(book, stress=_stress(), max_horizon_days=30)
    assert not result.is_breached
    assert result.survival_horizon_days == 30
    # CBC bottoms out at 0
    assert result.daily_ladder["running_cbc"].min() == pytest.approx(0.0, abs=1.0)


# --------------------------------------------------------- monotonicity
def test_higher_runoff_shortens_survival():
    book = _make_book(
        outflow_days=[10, 20, 30], outflow_amounts=[100_000, 100_000, 100_000],
        hqla_stock_amount=200_000.0,
        deposit_counterparty="wholesale",
    )
    low_runoff = _stress(wholesale_runoff=0.20)
    high_runoff = _stress(wholesale_runoff=0.80)
    r_low = compute_survival_horizon(book, stress=low_runoff, max_horizon_days=60)
    r_high = compute_survival_horizon(book, stress=high_runoff, max_horizon_days=60)
    assert r_high.survival_horizon_days <= r_low.survival_horizon_days


def test_larger_hqla_stock_extends_survival():
    book_small = _make_book(
        outflow_days=[10], outflow_amounts=[500_000],
        hqla_stock_amount=100_000.0,
        deposit_counterparty="wholesale",
    )
    book_large = _make_book(
        outflow_days=[10], outflow_amounts=[500_000],
        hqla_stock_amount=1_000_000.0,
        deposit_counterparty="wholesale",
    )
    r_small = compute_survival_horizon(book_small, stress=_stress(), max_horizon_days=60)
    r_large = compute_survival_horizon(book_large, stress=_stress(), max_horizon_days=60)
    assert r_large.survival_horizon_days >= r_small.survival_horizon_days


# --------------------------------------------------------- HQLA haircuts
def test_l2b_haircut_yields_less_cbc_than_l1():
    book_l1 = _make_book(outflow_days=[], outflow_amounts=[], hqla_stock_amount=1_000_000, hqla_type="Level1")
    book_l2b = _make_book(outflow_days=[], outflow_amounts=[], hqla_stock_amount=1_000_000, hqla_type="Level2B")
    r_l1 = compute_survival_horizon(book_l1, stress=_stress(), max_horizon_days=30)
    r_l2b = compute_survival_horizon(book_l2b, stress=_stress(), max_horizon_days=30)
    assert r_l1.initial_cbc > r_l2b.initial_cbc


# --------------------------------------------------------- combined stress
def test_combined_stress_is_at_least_as_severe_as_components():
    """Combined runoff >= max(idio, market) on the wholesale side ->
    survival horizon under combined <= each component."""
    book = _make_book(
        outflow_days=[5, 15, 30, 60], outflow_amounts=[200_000] * 4,
        inflow_days=[10, 25, 50], inflow_amounts=[80_000] * 3,
        hqla_stock_amount=500_000.0, hqla_type="Level2A",
        deposit_counterparty="wholesale",
    )
    r_idio = compute_survival_horizon(book, stress=LIQUIDITY_STRESS_SCENARIOS["idiosyncratic"], max_horizon_days=120)
    r_mkt = compute_survival_horizon(book, stress=LIQUIDITY_STRESS_SCENARIOS["market_wide"], max_horizon_days=120)
    r_comb = compute_survival_horizon(book, stress=LIQUIDITY_STRESS_SCENARIOS["combined"], max_horizon_days=120)
    assert r_comb.survival_horizon_days <= r_idio.survival_horizon_days
    assert r_comb.survival_horizon_days <= r_mkt.survival_horizon_days


# --------------------------------------------------------- NMD vs unstable
def test_nmd_flag_lowers_retail_runoff_factor():
    """Stable retail (NMD) runoff is lower than unstable retail, so survival
    under the same gross outflow is longer when the deposits are NMD-classified."""
    book_unstable = _make_book(
        outflow_days=[5], outflow_amounts=[500_000],
        hqla_stock_amount=50_000.0, hqla_type="Level1",
        deposit_counterparty="retail", is_nmd_flag=False,
    )
    book_nmd = _make_book(
        outflow_days=[5], outflow_amounts=[500_000],
        hqla_stock_amount=50_000.0, hqla_type="Level1",
        deposit_counterparty="retail", is_nmd_flag=True,
    )
    r_unstable = compute_survival_horizon(book_unstable, stress=_stress(), max_horizon_days=60)
    r_nmd = compute_survival_horizon(book_nmd, stress=_stress(), max_horizon_days=60)
    # Both should breach (50k CBC vs 500k * runoff), but NMD breach is later
    # because the stable runoff (5%) is smaller than unstable (10%).
    assert r_nmd.survival_horizon_days >= r_unstable.survival_horizon_days


# --------------------------------------------------------- inflow cap
def test_inflow_cap_does_not_bind_when_inflows_small():
    """When stressed inflows < outflows * cap, the cap doesn't bind and
    inflow_cap=0.75 produces the same ladder as inflow_cap=1.0."""
    # Stressed outflow = 100_000 * 0.40 (wholesale runoff) = 40_000
    # Stressed inflow  = 20_000 * 0.95 (1 - 0.05 asset_inflow_haircut) = 19_000
    # 19_000 < 40_000 * 0.75 = 30_000 → cap doesn't bind under either setting.
    book = _make_book(
        outflow_days=[10, 20], outflow_amounts=[100_000, 100_000],
        inflow_days=[10, 20], inflow_amounts=[20_000, 20_000],
        hqla_stock_amount=200_000.0,
        deposit_counterparty="wholesale",
    )
    r_capped = compute_survival_horizon(book, stress=_stress(), max_horizon_days=60, inflow_cap=0.75)
    r_uncapped = compute_survival_horizon(book, stress=_stress(), max_horizon_days=60, inflow_cap=1.00)
    np.testing.assert_allclose(
        r_capped.daily_ladder["running_cbc"].to_numpy(),
        r_uncapped.daily_ladder["running_cbc"].to_numpy(),
    )


def test_tighter_inflow_cap_never_extends_survival():
    """A tighter LCR cap can only reduce (or hold) recognised inflows, so the
    survival horizon under cap=0.5 is <= the horizon under cap=1.0."""
    book = _make_book(
        outflow_days=[10, 20, 30], outflow_amounts=[200_000] * 3,
        inflow_days=[10, 20, 30], inflow_amounts=[200_000] * 3,
        hqla_stock_amount=100_000.0,
        deposit_counterparty="wholesale",
    )
    r_tight = compute_survival_horizon(book, stress=_stress(), max_horizon_days=60, inflow_cap=0.25)
    r_loose = compute_survival_horizon(book, stress=_stress(), max_horizon_days=60, inflow_cap=1.00)
    assert r_tight.survival_horizon_days <= r_loose.survival_horizon_days


# --------------------------------------------------------- determinism + shape
def test_compute_is_deterministic_given_inputs():
    book = _make_book(
        outflow_days=[7, 14, 21], outflow_amounts=[100_000] * 3,
        hqla_stock_amount=150_000.0,
        deposit_counterparty="wholesale",
    )
    r1 = compute_survival_horizon(book, stress=_stress(), max_horizon_days=30)
    r2 = compute_survival_horizon(book, stress=_stress(), max_horizon_days=30)
    assert r1.survival_horizon_days == r2.survival_horizon_days
    assert r1.initial_cbc == r2.initial_cbc
    assert r1.peak_deficit == r2.peak_deficit


def test_ladder_has_one_row_per_day_in_horizon():
    book = _make_book(
        outflow_days=[5], outflow_amounts=[1_000],
        hqla_stock_amount=10_000,
    )
    horizon = 90
    r = compute_survival_horizon(book, stress=_stress(), max_horizon_days=horizon)
    assert len(r.daily_ladder) == horizon + 1
    assert r.daily_ladder["day_offset"].iloc[0] == 0
    assert r.daily_ladder["day_offset"].iloc[-1] == horizon


def test_running_cbc_is_monotone_in_outflow_amount():
    """Same shock geometry, more outflow -> running_cbc is at-or-below the
    smaller-outflow trajectory pointwise."""
    book_small = _make_book(outflow_days=[10], outflow_amounts=[10_000],
                            hqla_stock_amount=1_000_000.0,
                            deposit_counterparty="wholesale")
    book_big = _make_book(outflow_days=[10], outflow_amounts=[500_000],
                          hqla_stock_amount=1_000_000.0,
                          deposit_counterparty="wholesale")
    r_small = compute_survival_horizon(book_small, stress=_stress(), max_horizon_days=30)
    r_big = compute_survival_horizon(book_big, stress=_stress(), max_horizon_days=30)
    assert (r_big.daily_ladder["running_cbc"].to_numpy()
            <= r_small.daily_ladder["running_cbc"].to_numpy() + 1e-9).all()


# --------------------------------------------------------- missing columns
def test_refuses_when_required_columns_missing():
    book = pd.DataFrame({"amount": [1.0]})
    with pytest.raises(KeyError):
        compute_survival_horizon(book, stress=_stress())


def test_inflow_cap_out_of_range_raises():
    book = _make_book(outflow_days=[1], outflow_amounts=[1], hqla_stock_amount=1)
    with pytest.raises(ValueError):
        compute_survival_horizon(book, stress=_stress(), inflow_cap=1.5)


# --------------------------------------------------------- hypothesis
@given(
    outflow_amount=st.floats(min_value=1_000, max_value=1_000_000),
    cbc_amount=st.floats(min_value=0, max_value=10_000_000),
    runoff=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=20, deadline=None)
def test_survival_horizon_within_bounds(outflow_amount, cbc_amount, runoff):
    book = _make_book(
        outflow_days=[10], outflow_amounts=[outflow_amount],
        hqla_stock_amount=cbc_amount,
        deposit_counterparty="wholesale",
    )
    r = compute_survival_horizon(
        book, stress=_stress(wholesale_runoff=runoff), max_horizon_days=60,
    )
    assert 0 <= r.survival_horizon_days <= 60
