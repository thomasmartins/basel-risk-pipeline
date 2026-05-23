"""MC rate-path container and bulk discount factor utilities.

A `MCPathSet` wraps the (n_paths, n_steps+1) short-rate matrix produced by a
short-rate model. It carries `dt` so we can map step indices to year fractions
and discount cashflows by integrating along each path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basel_risk_engine.rate_models.vasicek import VasicekModel


@dataclass(frozen=True)
class MCPathSet:
    short_rates: np.ndarray  # (n_paths, n_steps + 1)
    dt: float
    model: VasicekModel

    @property
    def n_paths(self) -> int:
        return int(self.short_rates.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.short_rates.shape[1] - 1)

    @property
    def horizon_years(self) -> float:
        return self.n_steps * self.dt

    def discount_factors(self, taus: np.ndarray) -> np.ndarray:
        """Discount factors P(0, tau) for an array of tenors `taus` (in years),
        averaged across MC paths using the model's analytical bond price.

        Returns shape (len(taus),) — risk-neutral expectation under Vasicek.
        For each path we use its time-0 short rate (all paths start at r0, so
        analytic and MC agree at tau=0; for tau>0 we use the bond price formula
        with r0 — this is the path-mean by construction under risk-neutral pricing).
        """
        r0 = float(self.short_rates[0, 0])
        return self.model.bond_price(np.asarray(taus, dtype=np.float64), r0)

    def path_discount_factors(self, taus: np.ndarray) -> np.ndarray:
        """Per-path discount factors at the given tenors, evaluated from each
        path's short rate at time 0 (here: r0 — identical across paths) using
        the analytical Vasicek bond formula. Shape (n_paths, len(taus)).

        For Phase 2 we use t=0 short rates uniformly; Phase 2.1 will support
        forward valuation at intermediate steps (needed for proper NII paths).
        """
        taus = np.asarray(taus, dtype=np.float64)
        rates_t0 = self.short_rates[:, 0]
        # Outer product via broadcasting
        A, B = self.model._AB(taus)
        # P_i(tau) = A(tau) * exp(-B(tau) * r_i)
        return A[np.newaxis, :] * np.exp(-B[np.newaxis, :] * rates_t0[:, np.newaxis])


def simulate_paths(
    model: VasicekModel,
    *,
    n_paths: int,
    horizon_years: float,
    dt: float = 1 / 12,
    seed: int | None = None,
    antithetic: bool = True,
) -> MCPathSet:
    """Convenience builder. Steps = round(horizon_years / dt)."""
    n_steps = max(1, round(horizon_years / dt))
    rates = model.simulate(n_paths=n_paths, n_steps=n_steps, dt=dt, seed=seed, antithetic=antithetic)
    return MCPathSet(short_rates=rates, dt=dt, model=model)
