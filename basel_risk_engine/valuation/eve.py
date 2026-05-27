"""EVE engine: deterministic BCBS 368 shocks + MC distribution + supervisory outlier test.

EVE = Σ(asset PVs) − Σ(liability PVs). Phase 2.1c introduces type-aware pricing:

    bullet (default)
        Single-payment instrument repriced as `signed_amount * DF(behavioral tau)`.
        Used for short loans, bullet bonds (non-callable), deposits.

    level-payment mortgage (amortization_type == 'level')
        Full per-month interest + principal schedule, with CPR-driven prepayment.
        Each scheduled cashflow discounted at its own tenor on the input curve.

    callable bond (is_callable, non-mortgage)
        Straight bullet PV minus a Black-76 (HW1F closed-form) European call.
        Falls back to plain bullet pricing if the rate model isn't HW1F.

Sign convention:
    loans, mortgages, bonds (including callables) -> assets    (+1)
    deposits                                       -> liabilities (-1)

Two measures are produced:
    - Deterministic BCBS 368 ΔEVE under the six §132 shocks: fully optionality-aware.
    - MC ΔEVE distribution at a forward horizon: bullet-equivalent flat-table
      pricing (mortgage CPR schedule locked at the base curve; callable call
      value not path-dependent) — captures the linear/duration risk, with the
      convexity from optionality measured in the deterministic ΔEVE path.

EBA supervisory outlier test (EBA/GL/2022/14 §114, BCBS 368 §132):
    max over the six prescribed scenarios of |ΔEVE| / Tier1  ≤  15%
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from basel_risk_engine.behavioral.mortgage_cpr import (
    CPRParams,
    amortisation_schedule,
    cpr_curve,
    project_refi_rates,
)
from basel_risk_engine.rate_models.paths import MCPathSet, ShortRateModel
from basel_risk_engine.valuation.black76 import zbc_price
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

_ASSET_PRODUCTS = ("loan", "bond")


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


@dataclass(frozen=True)
class _FlatPaymentTable:
    """Flat per-payment representation used by MC bullet-equivalent pricing.

    Each row is one signed cashflow at tenor `tau_years`; mortgages are
    expanded into one row per scheduled month (CPR frozen at base curve).
    """
    taus: np.ndarray         # shape (M,)
    signed_amounts: np.ndarray  # shape (M,)


class EVEEngine:
    """Reprices a cashflow book and computes deterministic + MC EVE measures."""

    def __init__(
        self,
        base_curve: YieldCurve,
        rate_model: ShortRateModel | None = None,
        cpr_params: CPRParams | None = None,
    ):
        self.base_curve = base_curve
        self.rate_model = rate_model
        self.cpr_params = cpr_params or CPRParams()

    # ------------------------------------------------------------- type masks
    @staticmethod
    def _signs(products: pd.Series) -> np.ndarray:
        return np.where(products.isin(_ASSET_PRODUCTS), 1.0, -1.0)

    @staticmethod
    def _classify(cashflows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        if "amortization_type" in cashflows.columns:
            amort = cashflows["amortization_type"].fillna("bullet")
        else:
            amort = pd.Series("bullet", index=cashflows.index)
        if "is_callable" in cashflows.columns:
            is_call = cashflows["is_callable"].fillna(False).astype(bool)
        else:
            is_call = pd.Series(False, index=cashflows.index)
        is_mortgage = (amort == "level")
        is_callable_bond = is_call & ~is_mortgage
        is_bullet = ~is_mortgage & ~is_callable_bond
        return is_bullet, is_mortgage, is_callable_bond

    # ------------------------------------------------------- deterministic value
    def _value_bullets(self, sub: pd.DataFrame, curve: YieldCurve) -> float:
        if sub.empty:
            return 0.0
        taus = sub["behavioral_maturity_years"].to_numpy(dtype=np.float64)
        amounts = sub["amount"].to_numpy(dtype=np.float64)
        signs = self._signs(sub["product"])
        dfs = curve.discount_factor(taus)
        return float((signs * amounts * dfs).sum())

    def _value_mortgages(self, sub: pd.DataFrame, curve: YieldCurve) -> float:
        if sub.empty:
            return 0.0
        total = 0.0
        for r in sub.itertuples(index=False):
            term = int(r.term_months)
            c = float(r.customer_rate)
            N = float(r.amount)
            refi = project_refi_rates(curve, term)
            cpr = cpr_curve(c, refi, self.cpr_params)
            schedule = amortisation_schedule(N, c, term, cpr_per_month=cpr)
            taus = schedule["month"].to_numpy(dtype=np.float64) / 12.0
            cash = (
                schedule["interest"].to_numpy(dtype=np.float64)
                + schedule["total_principal"].to_numpy(dtype=np.float64)
            )
            dfs = curve.discount_factor(taus)
            # Mortgages are loans -> +1
            total += float((cash * dfs).sum())
        return total

    def _value_callables(self, sub: pd.DataFrame, curve: YieldCurve) -> float:
        if sub.empty:
            return 0.0
        # Straight bullet PV (bonds -> +1) under the input curve
        taus = sub["behavioral_maturity_years"].to_numpy(dtype=np.float64)
        amounts = sub["amount"].to_numpy(dtype=np.float64)
        dfs = curve.discount_factor(taus)
        straight_pv = float((amounts * dfs).sum())

        # Black-76 call deduction is HW1F-specific; fall back if not available.
        # Deferred import avoids a rate_models <-> valuation circular import.
        from basel_risk_engine.rate_models.hull_white import HullWhiteModel

        if not isinstance(self.rate_model, HullWhiteModel):
            return straight_pv
        a = self.rate_model.params.a
        sigma = self.rate_model.params.sigma
        call_total = 0.0
        for r in sub.itertuples(index=False):
            t_call = float(r.t_call_years)
            t_mat = float(r.behavioral_maturity_years)
            K_unit = float(r.call_strike_pct) / 100.0
            call_total += float(r.amount) * zbc_price(curve, a, sigma, t_call, t_mat, K_unit)
        return straight_pv - call_total

    def value(self, cashflows: pd.DataFrame, curve: YieldCurve) -> float:
        if "behavioral_maturity_years" not in cashflows.columns:
            raise KeyError(
                "EVEEngine.value expects `behavioral_maturity_years`; run the NMD "
                "overlay or set it equal to the contractual maturity first."
            )
        is_bullet, is_mortgage, is_callable_bond = self._classify(cashflows)
        return (
            self._value_bullets(cashflows[is_bullet], curve)
            + self._value_mortgages(cashflows[is_mortgage], curve)
            + self._value_callables(cashflows[is_callable_bond], curve)
        )

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

    # ------------------------------------------------------ MC flat-table prep
    def _build_flat_table(self, cashflows: pd.DataFrame) -> _FlatPaymentTable:
        is_bullet, is_mortgage, is_callable_bond = self._classify(cashflows)
        taus_list: list[np.ndarray] = []
        amts_list: list[np.ndarray] = []

        # Bullets and callables: one entry each at their behavioral maturity.
        # Callables are priced as straight bullets in the MC measure (call value
        # is captured separately in the deterministic BCBS 368 path).
        for mask in (is_bullet, is_callable_bond):
            sub = cashflows[mask]
            if sub.empty:
                continue
            taus_list.append(sub["behavioral_maturity_years"].to_numpy(dtype=np.float64))
            amts_list.append(self._signs(sub["product"]) * sub["amount"].to_numpy(dtype=np.float64))

        # Mortgages: expand each into its CPR-adjusted base-curve schedule.
        mortgages = cashflows[is_mortgage]
        for r in mortgages.itertuples(index=False):
            term = int(r.term_months)
            c = float(r.customer_rate)
            N = float(r.amount)
            refi = project_refi_rates(self.base_curve, term)
            cpr = cpr_curve(c, refi, self.cpr_params)
            schedule = amortisation_schedule(N, c, term, cpr_per_month=cpr)
            taus_list.append(schedule["month"].to_numpy(dtype=np.float64) / 12.0)
            cash = (
                schedule["interest"].to_numpy(dtype=np.float64)
                + schedule["total_principal"].to_numpy(dtype=np.float64)
            )
            # Mortgages are loans -> +1
            amts_list.append(cash)

        if not taus_list:
            return _FlatPaymentTable(taus=np.array([]), signed_amounts=np.array([]))
        return _FlatPaymentTable(
            taus=np.concatenate(taus_list),
            signed_amounts=np.concatenate(amts_list),
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
            1. take simulated short rate r_h at t = forward_horizon_years
            2. discount the (book-level) flat per-payment table via the model's
               instantaneous-shock formula  P(0, tau; r_h) = A(tau) exp(-B(tau) r_h)
            3. subtract the base-curve PV of the same table
        Returns shape (n_paths,) of ΔEVE values.

        The flat table is built once at the base curve (CPR for mortgages is
        therefore frozen at baseline, and the callable bond call value is not
        repriced per path). The deterministic BCBS 368 ΔEVE captures the full
        optionality response.
        """
        if self.rate_model is None:
            raise RuntimeError("MC distribution requires a calibrated short-rate model.")

        table = self._build_flat_table(cashflows)
        if table.taus.size == 0:
            return np.zeros(paths.n_paths)

        step = int(round(forward_horizon_years / paths.dt))
        step = min(step, paths.n_steps)
        r_h = paths.short_rates[:, step]

        A, B = self.rate_model._AB(table.taus)
        # baseline PV under the model's A,B at r=r_0 == base curve by HW1F construction
        baseline_pv = float((table.signed_amounts * (A * np.exp(-B * self.rate_model.params.r0))).sum())

        discount_per_path = A[None, :] * np.exp(-B[None, :] * r_h[:, None])
        per_path_pv = (table.signed_amounts[None, :] * discount_per_path).sum(axis=1)
        return per_path_pv - baseline_pv
