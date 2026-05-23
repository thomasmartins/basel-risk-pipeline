"""Non-Maturity Deposit (NMD) behavioral overlay.

NMDs (current accounts, demand deposits) have no contractual maturity but
behave with strong stickiness. EBA Guidelines on IRRBB (EBA/GL/2022/14) and
BCBS 368 both require institutions to model NMDs with an internal "behavioral"
repricing profile rather than the contractual one.

Phase 2 model — parametric and deliberately simple:
    - A `stable_core_pct` fraction of each NMD is "core" and is repriced at a
      long behavioral maturity (e.g. 5y); the residual is "non-core" and uses
      contractual maturity.
    - The volume-weighted average maturity becomes the cashflow's effective
      repricing tenor for EVE / NII purposes.
    - `deposit_beta` is the pass-through of market-rate changes onto the
      deposit rate (consumed by the NII engine, not EVE directly).

Phase 2.1 will add a runoff schedule (exponential decay with half-life) and a
proper rate-sensitive deposit-rate function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class NMDParams(BaseModel):
    """Frozen NMD modelling parameters."""

    model_config = ConfigDict(frozen=True)

    stable_core_pct: float = Field(default=0.70, ge=0.0, le=1.0)
    core_behavioral_maturity_yrs: float = Field(default=5.0, gt=0.0, le=30.0)
    deposit_beta: float = Field(default=0.40, ge=0.0, le=1.0)
    short_maturity_threshold_days: int = Field(default=365, gt=0)


def apply_nmd_overlay(
    cashflows: pd.DataFrame,
    params: NMDParams,
    *,
    product_col: str = "product",
    maturity_days_col: str = "maturity_days",
) -> pd.DataFrame:
    """Add a `behavioral_maturity_years` column to a cashflow frame.

    Non-NMD rows: behavioral = contractual.
    NMD rows (deposits with contractual maturity ≤ threshold):
        behavioral = core_pct * core_yrs + (1 - core_pct) * contractual_yrs
    """
    if maturity_days_col not in cashflows.columns:
        raise KeyError(f"Cashflow frame is missing required column '{maturity_days_col}'.")
    df = cashflows.copy()

    contractual_yrs = df[maturity_days_col].astype(float) / 365.0
    is_nmd = (
        (df[product_col] == "deposit")
        & (df[maturity_days_col].astype(float) <= params.short_maturity_threshold_days)
    )

    behavioral_yrs = np.where(
        is_nmd,
        params.stable_core_pct * params.core_behavioral_maturity_yrs
        + (1.0 - params.stable_core_pct) * contractual_yrs,
        contractual_yrs,
    )
    df["behavioral_maturity_years"] = behavioral_yrs
    df["is_nmd"] = is_nmd
    return df
