"""MC rate-path container and bulk discount factor utilities.

A `MCPathSet` wraps the (n_paths, n_steps+1) short-rate matrix produced by a
short-rate model. It carries `dt` so we can map step indices to year fractions
and discount cashflows by integrating along each path.

The container is model-agnostic: any object satisfying the `ShortRateModel`
protocol (i.e. exposing `_AB(tau)` and `bond_price(tau, r)` with t=0 semantics)
plugs in — Vasicek 1F and Hull-White 1F both qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ShortRateModel(Protocol):
    """Minimal interface required by MCPathSet and EVEEngine.

    Both methods are evaluated at valuation date t = 0; HW1F's curve-aware
    A(0, tau) is absorbed into _AB on the model side. simulate() is consumed
    via simulate_paths() below.
    """

    def _AB(self, tau): ...
    def bond_price(self, tau, r): ...
    def simulate(self, n_paths, n_steps, dt, *, seed=None, antithetic=True): ...


@dataclass(frozen=True)
class MCPathSet:
    short_rates: np.ndarray  # (n_paths, n_steps + 1)
    dt: float
    model: ShortRateModel

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
        """Discount factors P(0, tau) evaluated from the time-0 short rate using
        the model's analytical bond formula.

        For Vasicek this is the analytic risk-neutral expectation under the
        calibrated parameters; for Hull-White it is the input market curve
        evaluated at the requested tenors (since A(0,τ)·exp(-B(0,τ)·r_0) = P^M(0,τ)).
        """
        r0 = float(self.short_rates[0, 0])
        return self.model.bond_price(np.asarray(taus, dtype=np.float64), r0)

    def path_discount_factors(self, taus: np.ndarray) -> np.ndarray:
        """Per-path discount factors at the given tenors, evaluated from each
        path's short rate at time 0 (here: r0 — identical across paths) using
        the analytical bond formula. Shape (n_paths, len(taus)).

        For Phase 2 we use t=0 short rates uniformly; Phase 2.1+ will support
        forward valuation at intermediate steps (needed for proper NII paths).
        """
        taus = np.asarray(taus, dtype=np.float64)
        rates_t0 = self.short_rates[:, 0]
        A, B = self.model._AB(taus)
        return A[np.newaxis, :] * np.exp(-B[np.newaxis, :] * rates_t0[:, np.newaxis])


def simulate_paths(
    model: ShortRateModel,
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
