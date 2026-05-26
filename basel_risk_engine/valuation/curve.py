"""Zero-coupon yield curve with per-bucket shift support.

Internally stores a tenor → zero-yield grid; yields at arbitrary tenors are
linearly interpolated (yields, not discount factors — closer to practice).

Bucket shifts use the EBA-style year buckets defined in
`basel_risk_engine.scenarios.bcbs368`. A shift dict like `{"0-1y": 200, ...}`
adds the per-bucket bps to the zero yield at every tenor that falls inside
that bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Same bucket grid as in compute.py / dbt
EBA_BUCKETS = ("0-1y", "1-3y", "3-5y", "5-10y", "10y+")
_BUCKET_UPPER_BOUND_YRS = {
    "0-1y": 1.0,
    "1-3y": 3.0,
    "3-5y": 5.0,
    "5-10y": 10.0,
    "10y+": float("inf"),
}


def _bucket_for(tau_years: float) -> str:
    for b in EBA_BUCKETS:
        if tau_years <= _BUCKET_UPPER_BOUND_YRS[b]:
            return b
    return "10y+"


@dataclass(frozen=True)
class YieldCurve:
    tenors_years: np.ndarray
    zero_yields: np.ndarray

    def __post_init__(self):
        if self.tenors_years.shape != self.zero_yields.shape:
            raise ValueError("tenors and yields must have same shape")
        if not np.all(np.diff(self.tenors_years) > 0):
            raise ValueError("tenors must be strictly increasing")

    def yield_at(self, taus: float | np.ndarray) -> np.ndarray:
        taus = np.asarray(taus, dtype=np.float64)
        return np.interp(taus, self.tenors_years, self.zero_yields)

    def discount_factor(self, taus: float | np.ndarray) -> np.ndarray:
        taus = np.asarray(taus, dtype=np.float64)
        return np.exp(-self.yield_at(taus) * taus)

    def forward_rate(self, taus: float | np.ndarray) -> np.ndarray:
        """Instantaneous forward rate f(0, τ) under linear-yield interpolation.

        With y(τ) linearly interpolated on (tenors, yields),
            log P(0, τ) = -y(τ) · τ
            f(0, τ)     = -d/dτ log P(0, τ) = y(τ) + τ · dy/dτ

        dy/dτ is piecewise-constant within buckets; jumps at grid points are
        absorbed by evaluating the slope of the right-hand segment (consistent
        with np.interp's left-closed convention).
        """
        taus = np.atleast_1d(np.asarray(taus, dtype=np.float64))
        y = np.interp(taus, self.tenors_years, self.zero_yields)
        # Right-side index for each τ; clip at the last segment
        idx = np.searchsorted(self.tenors_years, taus, side="right")
        idx = np.clip(idx, 1, len(self.tenors_years) - 1)
        slope = (self.zero_yields[idx] - self.zero_yields[idx - 1]) / (
            self.tenors_years[idx] - self.tenors_years[idx - 1]
        )
        # Below the first grid point treat dy/dτ as zero (flat extrapolation)
        below = taus < self.tenors_years[0]
        slope = np.where(below, 0.0, slope)
        return y + taus * slope

    def shifted(self, shifts_bps: dict[str, float]) -> "YieldCurve":
        """Apply per-bucket bps shifts to every grid tenor."""
        new_yields = self.zero_yields.copy()
        for i, tau in enumerate(self.tenors_years):
            b = _bucket_for(float(tau))
            new_yields[i] = new_yields[i] + shifts_bps.get(b, 0.0) / 10_000.0
        return YieldCurve(tenors_years=self.tenors_years.copy(), zero_yields=new_yields)

    def parallel_shifted(self, bps: float) -> "YieldCurve":
        new_yields = self.zero_yields + bps / 10_000.0
        return YieldCurve(tenors_years=self.tenors_years.copy(), zero_yields=new_yields)
