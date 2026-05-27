"""ALMM-style liquidity survival horizon.

Under a liquidity stress scenario, the bank's survival horizon is the number
of days until cumulative net outflows would exhaust its counterbalancing
capacity (CBC) — the stock of monetisable high-quality liquid assets (HQLA)
plus subsequent net cash receipts.

Mechanics (per scenario × stress):

    1. HQLA stock t=0: the asset-side HQLA inventory after stressed haircuts.
       Level1 / Level2A / Level2B carry distinct haircuts that widen in
       market-wide and combined stresses (EBA LCR Annex II baseline; stressed
       haircuts taken slightly above LCR floors per common practice).

    2. Maturity ladder: non-HQLA cashflows are bucketed by day_offset. Stress
       overlays are applied row-by-row:

           outflow rows (deposits, wholesale)     -> stressed by runoff %
           inflow rows  (loan / non-HQLA inflows) -> haircut for credit losses

       HQLA inflows are *removed* from the ladder (they're already in the t=0
       stock — we'd realise them by selling, not by holding to maturity).

    3. LCR-style inflow cap: per-day capped inflows = min(inflow,
       outflow * inflow_cap). Defaults to 75% (LCR Article 33 §4).

    4. Walk forward day-by-day:
           running_cbc[d] = initial_cbc + cumulative_net_cashflow(0..d)
       Survival horizon = smallest d with running_cbc[d] < 0; or the maximum
       horizon if it never breaches (interpreted as "≥ max_horizon_days").

Three preset stress scenarios live in `LIQUIDITY_STRESS_SCENARIOS`:

    idiosyncratic   bank-specific shock (downgrade, retail flight)
    market_wide     no access to wholesale funding; HQLA haircuts widen
    combined        both simultaneously; aligns with severe ICAAP/ILAAP runs

These are *configurations*, not market data — they live in code, not in the
warehouse (same pattern as `NMDParams`, `CPRParams`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

StressName = Literal["idiosyncratic", "market_wide", "combined"]


class LiquidityStressParams(BaseModel):
    """Runoff factors and HQLA haircuts for an ALMM-style survival horizon."""

    model_config = ConfigDict(frozen=True)

    # Deposit runoff: fraction of contractual outflow that actually leaves
    retail_stable_runoff: float = Field(default=0.05, ge=0.0, le=1.0)
    retail_unstable_runoff: float = Field(default=0.10, ge=0.0, le=1.0)
    wholesale_runoff: float = Field(default=0.40, ge=0.0, le=1.0)

    # Inflow side: credit losses on scheduled loan / non-HQLA-bond repayments
    asset_inflow_haircut: float = Field(default=0.05, ge=0.0, le=1.0)

    # Standing HQLA haircuts under this stress (above LCR baselines in market /
    # combined stresses to reflect fire-sale conditions).
    hqla_haircut_l1: float = Field(default=0.00, ge=0.0, le=1.0)
    hqla_haircut_l2a: float = Field(default=0.15, ge=0.0, le=1.0)
    hqla_haircut_l2b: float = Field(default=0.50, ge=0.0, le=1.0)


# Preset stress scenarios. Names are picked up by run.py and the dashboard.
LIQUIDITY_STRESS_SCENARIOS: dict[StressName, LiquidityStressParams] = {
    "idiosyncratic": LiquidityStressParams(
        retail_stable_runoff=0.05,
        retail_unstable_runoff=0.10,
        wholesale_runoff=0.40,
        asset_inflow_haircut=0.05,
        hqla_haircut_l1=0.00,
        hqla_haircut_l2a=0.15,
        hqla_haircut_l2b=0.50,
    ),
    "market_wide": LiquidityStressParams(
        retail_stable_runoff=0.05,
        retail_unstable_runoff=0.10,
        wholesale_runoff=1.00,
        asset_inflow_haircut=0.10,
        hqla_haircut_l1=0.00,
        hqla_haircut_l2a=0.25,
        hqla_haircut_l2b=0.60,
    ),
    "combined": LiquidityStressParams(
        retail_stable_runoff=0.15,
        retail_unstable_runoff=0.30,
        wholesale_runoff=1.00,
        asset_inflow_haircut=0.15,
        hqla_haircut_l1=0.05,
        hqla_haircut_l2a=0.30,
        hqla_haircut_l2b=0.65,
    ),
}


@dataclass(frozen=True)
class SurvivalResult:
    stress_name: str
    initial_cbc: float
    survival_horizon_days: int     # max_horizon_days if never breached
    is_breached: bool
    peak_deficit: float            # most negative running_cbc; 0 if never goes negative
    daily_ladder: pd.DataFrame     # full per-day trajectory


def _hqla_haircut(hqla_type: pd.Series, p: LiquidityStressParams) -> np.ndarray:
    """Map hqla_type strings to the stress haircut. Non-HQLA -> 100% (won't be
    counted as CBC because we filter on hqla_type != 'None' upstream)."""
    return np.select(
        condlist=[
            hqla_type == "Level1",
            hqla_type == "Level2A",
            hqla_type == "Level2B",
        ],
        choicelist=[
            p.hqla_haircut_l1,
            p.hqla_haircut_l2a,
            p.hqla_haircut_l2b,
        ],
        default=1.0,
    )


def _outflow_runoff(cashflows: pd.DataFrame, p: LiquidityStressParams) -> np.ndarray:
    """Per-row outflow runoff factor (fraction of the contractual outflow that
    actually leaves under stress).

    Retail deposits split into stable / unstable by `is_nmd` (NMDs are the
    "stable core" proxy in the synthetic data); wholesale captures everything
    counterparty='wholesale'. Non-deposit outflows pass through unchanged
    (e.g. bond redemptions are contractual).
    """
    n = len(cashflows)
    factor = np.ones(n, dtype=np.float64)
    is_deposit = (cashflows["product"] == "deposit").to_numpy()
    is_retail = (cashflows["counterparty"] == "retail").to_numpy()
    is_wholesale = (cashflows["counterparty"] == "wholesale").to_numpy()
    is_nmd = cashflows.get("is_nmd", pd.Series(False, index=cashflows.index)).fillna(False).astype(bool).to_numpy()

    retail_dep = is_deposit & is_retail
    wholesale_dep = is_deposit & is_wholesale
    factor = np.where(retail_dep & is_nmd, p.retail_stable_runoff, factor)
    factor = np.where(retail_dep & ~is_nmd, p.retail_unstable_runoff, factor)
    factor = np.where(wholesale_dep, p.wholesale_runoff, factor)
    return factor


def compute_survival_horizon(
    cashflows: pd.DataFrame,
    *,
    stress: LiquidityStressParams,
    stress_name: str = "custom",
    max_horizon_days: int = 365,
    inflow_cap: float = 0.75,
) -> SurvivalResult:
    """ALMM-style survival horizon under one stress configuration.

    Required cashflow columns: maturity_days, amount, direction, product,
    counterparty, hqla_type. `is_nmd` is optional (treated as False if missing).
    """
    required = {"maturity_days", "amount", "direction", "product", "counterparty", "hqla_type"}
    missing = required - set(cashflows.columns)
    if missing:
        raise KeyError(f"compute_survival_horizon missing columns: {sorted(missing)}")
    if not 0.0 <= inflow_cap <= 1.0:
        raise ValueError(f"inflow_cap must be in [0, 1]; got {inflow_cap}")

    df = cashflows.copy()
    df["day"] = df["maturity_days"].astype(int)

    # --- t=0 HQLA stock ---------------------------------------------------
    # HQLA inventory is realised at t=0 (we'd sell under stress), so its
    # maturity is irrelevant — long-dated Level1 bonds still contribute to CBC.
    is_hqla_inflow = (df["direction"] == "inflow") & (df["hqla_type"].isin(["Level1", "Level2A", "Level2B"]))
    hqla = df[is_hqla_inflow]
    if not hqla.empty:
        haircuts = _hqla_haircut(hqla["hqla_type"], stress)
        initial_cbc = float((hqla["amount"].to_numpy(dtype=np.float64) * (1.0 - haircuts)).sum())
    else:
        initial_cbc = 0.0

    # --- ladder rows: everything that isn't HQLA AND matures within horizon
    # Cashflows past the horizon are excluded — they have no impact on whether
    # the bank survives the next `max_horizon_days` days. Clipping them to day
    # max_horizon would pile a fake mass at the right edge.
    within_horizon = df["day"] <= max_horizon_days
    ladder_src = df[~is_hqla_inflow & within_horizon].copy()
    ladder_src["day"] = ladder_src["day"].clip(lower=0)

    # Outflows
    out_mask = (ladder_src["direction"] == "outflow")
    out_factor = np.zeros(len(ladder_src), dtype=np.float64)
    if out_mask.any():
        out_factor[out_mask.to_numpy()] = _outflow_runoff(ladder_src[out_mask], stress)
    ladder_src["stressed_outflow"] = ladder_src["amount"].to_numpy(dtype=np.float64) * out_factor

    # Inflows (loan / non-HQLA bond repayments) with credit haircut
    in_mask = (ladder_src["direction"] == "inflow")
    in_factor = np.zeros(len(ladder_src), dtype=np.float64)
    in_factor[in_mask.to_numpy()] = 1.0 - stress.asset_inflow_haircut
    ladder_src["stressed_inflow"] = ladder_src["amount"].to_numpy(dtype=np.float64) * in_factor

    # Aggregate by day
    grouped = (
        ladder_src.groupby("day", as_index=False)
        .agg(stressed_inflow=("stressed_inflow", "sum"),
             stressed_outflow=("stressed_outflow", "sum"))
    )

    # Fill in a complete day grid 0..max_horizon_days
    grid = pd.DataFrame({"day": np.arange(max_horizon_days + 1)})
    ladder = grid.merge(grouped, on="day", how="left").fillna(0.0)

    # LCR-style inflow cap
    ladder["capped_inflow"] = np.minimum(
        ladder["stressed_inflow"].to_numpy(),
        ladder["stressed_outflow"].to_numpy() * inflow_cap,
    )
    ladder["net_cashflow"] = ladder["capped_inflow"] - ladder["stressed_outflow"]
    ladder["cumulative_net"] = ladder["net_cashflow"].cumsum()
    ladder["running_cbc"] = initial_cbc + ladder["cumulative_net"]

    # Survival horizon: first day running_cbc < 0
    breaches = ladder.index[ladder["running_cbc"] < 0].to_numpy()
    if breaches.size > 0:
        survival_horizon_days = int(ladder.loc[breaches[0], "day"])
        is_breached = True
    else:
        survival_horizon_days = int(max_horizon_days)
        is_breached = False

    peak_deficit = float(min(0.0, ladder["running_cbc"].min()))

    ladder = ladder.rename(columns={"day": "day_offset"})
    ladder["stress_name"] = stress_name

    return SurvivalResult(
        stress_name=stress_name,
        initial_cbc=initial_cbc,
        survival_horizon_days=survival_horizon_days,
        is_breached=is_breached,
        peak_deficit=peak_deficit,
        daily_ladder=ladder,
    )
