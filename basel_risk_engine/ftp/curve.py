"""Funds Transfer Pricing internal curve.

An FTP curve is the matched-tenor *internal* funding rate that ALM/treasury
charges (assets) or credits (liabilities) to a business unit for each
contractual cashflow. It is constructed as

    ftp_yield(tau) = wholesale_yield(tau) + liquidity_premium(tau)

where the wholesale curve is the bank's external funding curve (proxied here
by the input zero curve), and the liquidity premium is a tenor-dependent
add-on that compensates treasury for the maturity transformation and
contingent-liquidity risk it bears on behalf of the business units.

Real ALM stacks add further components (basis-cost, capital cost, expected
loss) — the LP add-on here is the headline one and the place where Phase 2.2+
extensions can hook in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from basel_risk_engine.valuation.curve import YieldCurve


@dataclass(frozen=True)
class LiquidityPremiumSchedule:
    """Per-tenor liquidity premium add-on in basis points.

    Linearly interpolated. Constant extrapolation beyond the first/last grid
    point (i.e. flat tails) — same convention as the YieldCurve interpolation.
    """

    tenors_years: np.ndarray
    lp_bps: np.ndarray

    def __post_init__(self):
        if self.tenors_years.shape != self.lp_bps.shape:
            raise ValueError("tenors and lp_bps must have same shape")
        if not np.all(np.diff(self.tenors_years) > 0):
            raise ValueError("tenors must be strictly increasing")

    def at(self, taus: float | np.ndarray) -> np.ndarray:
        taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        return np.interp(taus, self.tenors_years, self.lp_bps)

    @classmethod
    def flat(cls, tenors_years: np.ndarray, lp_bps: float) -> "LiquidityPremiumSchedule":
        return cls(
            tenors_years=np.asarray(tenors_years, dtype=np.float64),
            lp_bps=np.full(len(tenors_years), float(lp_bps), dtype=np.float64),
        )

    @classmethod
    def zero(cls, tenors_years: np.ndarray) -> "LiquidityPremiumSchedule":
        return cls.flat(tenors_years, 0.0)


@dataclass(frozen=True)
class FTPCurve:
    """Internal FTP curve = wholesale base curve + liquidity premium add-on."""

    base_curve: YieldCurve
    lp_schedule: LiquidityPremiumSchedule

    def ftp_yield(self, taus: float | np.ndarray) -> np.ndarray:
        """ftp_yield(τ) = base_yield(τ) + lp(τ) / 1e4."""
        taus_arr = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        base = self.base_curve.yield_at(taus_arr)
        lp = self.lp_schedule.at(taus_arr) / 10_000.0
        return base + lp

    def discount_factor(self, taus: float | np.ndarray) -> np.ndarray:
        taus_arr = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        return np.exp(-self.ftp_yield(taus_arr) * taus_arr)

    def overnight_funding_rate(self) -> float:
        """Wholesale O/N rate proxy: instantaneous forward at τ → 0 + LP at τ → 0.

        Used as the reference 'cost of funding' in the funding-margin component
        of the NII attribution.
        """
        on_tau = np.array([1e-6])
        return float(self.ftp_yield(on_tau)[0])

    def to_grid_frame(self) -> dict[str, np.ndarray]:
        """Return the FTP yield on the union of (base curve, LP schedule) tenor grids."""
        tenors = np.unique(np.concatenate([self.base_curve.tenors_years, self.lp_schedule.tenors_years]))
        return {
            "tenor_years": tenors,
            "base_yield": self.base_curve.yield_at(tenors),
            "lp_bps": self.lp_schedule.at(tenors),
            "ftp_yield": self.ftp_yield(tenors),
        }
