"""NII engine: forward-NII distribution under MC rate paths, with behavioral repricing.

For each MC path p and horizon h (in years) we compute the path-averaged
short rate r̄_p(h) and define a one-shot NII shock proxy:

    ΔNII_p(h) = Σ_i  sign_i · amount_i · beta_i · 1[m_i ≤ h] · (r̄_p(h) − r_0) · h

where
    sign_i  = +1 for loans / bonds, −1 for deposits
    beta_i  = deposit_beta if row is an NMD, else 1.0  (full pass-through)
    m_i     = behavioural maturity (years) — comes from the NMD overlay
    r_0     = baseline short rate (time-0 of the MC simulation)

It's an approximation — a true NII engine would walk a cashflow ladder under
each path's evolving curve and reprice at each step. But this captures the
critical mechanics (repricing gap × rate shift, behavioural NMD dampening,
horizon scaling) and yields a usable distribution for the fan chart and the
∆NII-at-risk metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from basel_risk_engine.behavioral.nmd import NMDParams
from basel_risk_engine.rate_models.paths import MCPathSet


def compute_nii_paths(
    cashflows: pd.DataFrame,
    paths: MCPathSet,
    *,
    nmd: NMDParams,
    horizons_months: tuple[int, ...] = (12, 24, 36),
) -> pd.DataFrame:
    """Returns a long DataFrame: scenario_id (if present), horizon_months, path_id, delta_nii."""
    required = {"behavioral_maturity_years", "amount", "product", "is_nmd"}
    missing = required - set(cashflows.columns)
    if missing:
        raise KeyError(f"NII engine missing columns: {sorted(missing)}")

    products = cashflows["product"]
    signs = np.where(products.isin(["loan", "bond"]), 1.0, -1.0)
    amounts = cashflows["amount"].to_numpy(dtype=np.float64)
    behavioral = cashflows["behavioral_maturity_years"].to_numpy(dtype=np.float64)
    is_nmd = cashflows["is_nmd"].to_numpy(dtype=bool)
    beta_factor = np.where(is_nmd, nmd.deposit_beta, 1.0)

    r0 = float(paths.short_rates[0, 0])

    rows: list[dict] = []
    for h_months in horizons_months:
        h_yrs = h_months / 12.0
        step_h = max(1, min(paths.n_steps, int(round(h_yrs / paths.dt))))

        # Path-average short rate over [0, h]
        avg_r = paths.short_rates[:, : step_h + 1].mean(axis=1)
        delta_r = avg_r - r0  # (n_paths,)

        reprices = behavioral <= h_yrs
        gap = float((signs * amounts * beta_factor * reprices).sum())
        # Per-path ΔNII over the horizon
        delta_nii = gap * delta_r * h_yrs

        for path_id, dn in enumerate(delta_nii):
            rows.append(
                {
                    "horizon_months": h_months,
                    "path_id": int(path_id),
                    "delta_nii": float(dn),
                    "repricing_gap": gap,
                    "avg_short_rate": float(avg_r[path_id]),
                }
            )
    return pd.DataFrame(rows)
