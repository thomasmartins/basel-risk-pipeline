"""EVE engine: deterministic BCBS 368 shocks + MC distribution + supervisory outlier test.

EVE = Σ(asset PVs) − Σ(liability PVs), where each instrument is repriced by
discounting its (post-behavioural) cashflows under the chosen curve.

We treat one synthetic instrument per cashflow row, valued as a bullet at
`behavioral_maturity_years` from valuation date with `amount` notional. This
is a simplification consistent with the data shape; real ALM uses full
amortisation schedules.

Sign convention:
    loans, bonds       -> assets    (+1)
    deposits           -> liabilities (-1)

EBA supervisory outlier test (EBA/GL/2022/14 §114, BCBS 368 §132):
    max over the six prescribed scenarios of |ΔEVE| / Tier1  ≤  15%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from basel_risk_engine.rate_models.paths import MCPathSet
from basel_risk_engine.rate_models.vasicek import VasicekModel
from basel_risk_engine.valuation.curve import EBA_BUCKETS, YieldCurve

# BCBS 368 §132 six prescribed scenarios. Bucket-resolved approximation of the
# continuous parameterisation in Annex 2 (parallel ±200bps, short ±250bps with
# decay, steepener/flattener with short -65 / long +90 etc.).
BCBS368_SHOCKS_BPS: dict[str, list[float]] = {
    "Parallel up":     [200, 200, 200, 200, 200],
    "Parallel down":   [-200, -200, -200, -200, -200],
    "Short rate up":   [250, 180, 100, 40, 0],
    "Short rate down": [-250, -180, -100, -40, 0],
    "Steepener":       [-65, -30, 30, 70, 90],
    "Flattener":       [90, 50, 0, -40, -65],
}


def _shock_dict(bucket_shifts: list[float]) -> dict[str, float]:
    return dict(zip(EBA_BUCKETS, bucket_shifts))


class EVEScenarioResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    scenario: str
    delta_eve: float


@dataclass(frozen=True)
class SupervisoryOutlierResult:
    worst_scenario: str
    worst_delta_eve: float
    tier1_capital: float
    ratio: float           # |ΔEVE| / Tier1
    breach: bool           # ratio > 0.15
    per_scenario: dict[str, float]
    distributional_99: float | None = None   # 99th percentile of |ΔEVE| from MC, if computed


class EVEEngine:
    """Reprices a cashflow book and computes deterministic + MC EVE measures."""

    def __init__(self, base_curve: YieldCurve, vasicek_model: VasicekModel | None = None):
        self.base_curve = base_curve
        self.vasicek_model = vasicek_model

    # --------------------------------------------------------------- pricing
    @staticmethod
    def _signs(products: pd.Series) -> np.ndarray:
        return np.where(products.isin(["loan", "bond"]), 1.0, -1.0)

    def value(self, cashflows: pd.DataFrame, curve: YieldCurve) -> float:
        if "behavioral_maturity_years" not in cashflows.columns:
            raise KeyError(
                "EVEEngine.value expects `behavioral_maturity_years`; run the NMD "
                "overlay or set it equal to the contractual maturity first."
            )
        taus = cashflows["behavioral_maturity_years"].to_numpy(dtype=np.float64)
        amounts = cashflows["amount"].to_numpy(dtype=np.float64)
        signs = self._signs(cashflows["product"])
        dfs = curve.discount_factor(taus)
        return float((signs * amounts * dfs).sum())

    # --------------------------------------------------------- BCBS 368 scenarios
    def bcbs368(self, cashflows: pd.DataFrame) -> list[EVEScenarioResult]:
        baseline_eve = self.value(cashflows, self.base_curve)
        results: list[EVEScenarioResult] = []
        for name, shifts in BCBS368_SHOCKS_BPS.items():
            shocked = self.base_curve.shifted(_shock_dict(shifts))
            delta = self.value(cashflows, shocked) - baseline_eve
            results.append(EVEScenarioResult(scenario=name, delta_eve=delta))
        return results

    # --------------------------------------------------------- supervisory test
    def supervisory_outlier_test(
        self,
        cashflows: pd.DataFrame,
        tier1_capital: float,
        threshold: float = 0.15,
        *,
        distributional_paths: MCPathSet | None = None,
    ) -> SupervisoryOutlierResult:
        scenario_results = self.bcbs368(cashflows)
        per_scenario = {r.scenario: r.delta_eve for r in scenario_results}
        worst = max(scenario_results, key=lambda r: abs(r.delta_eve))
        ratio = abs(worst.delta_eve) / tier1_capital if tier1_capital > 0 else 0.0

        dist_99: float | None = None
        if distributional_paths is not None:
            dist = self.mc_distribution(cashflows, distributional_paths)
            dist_99 = float(np.percentile(np.abs(dist), 99))

        return SupervisoryOutlierResult(
            worst_scenario=worst.scenario,
            worst_delta_eve=worst.delta_eve,
            tier1_capital=tier1_capital,
            ratio=ratio,
            breach=ratio > threshold,
            per_scenario=per_scenario,
            distributional_99=dist_99,
        )

    # --------------------------------------------------------- MC distribution
    def mc_distribution(
        self,
        cashflows: pd.DataFrame,
        paths: MCPathSet,
        *,
        forward_horizon_years: float = 1.0,
    ) -> np.ndarray:
        """Distribution of ΔEVE at a forward horizon under MC short-rate paths.

        For each path:
            1. find the simulated short rate r_h at t = horizon_years
            2. build a Vasicek-implied curve from r_h
            3. value the book under that curve, take the diff vs the baseline EVE
        Returns shape (n_paths,) of ΔEVE values.
        """
        if self.vasicek_model is None:
            raise RuntimeError("MC distribution requires a calibrated Vasicek model.")

        baseline_eve = self.value(cashflows, self.base_curve)
        step = int(round(forward_horizon_years / paths.dt))
        step = min(step, paths.n_steps)
        r_h = paths.short_rates[:, step]

        taus = cashflows["behavioral_maturity_years"].to_numpy(dtype=np.float64)
        amounts = cashflows["amount"].to_numpy(dtype=np.float64)
        signs = self._signs(cashflows["product"])

        # Vasicek-implied yields for each path at each tau
        A, B = self.vasicek_model._AB(taus)
        # discount_per_path[i, j] = A_j * exp(-B_j * r_h_i)
        discount_per_path = A[np.newaxis, :] * np.exp(-B[np.newaxis, :] * r_h[:, np.newaxis])
        per_path_pv = (signs * amounts) * discount_per_path  # (n_paths, n_cashflows)
        per_path_eve = per_path_pv.sum(axis=1)
        return per_path_eve - baseline_eve
