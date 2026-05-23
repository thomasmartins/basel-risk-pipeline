"""Vasicek one-factor short-rate model.

SDE: dr_t = kappa (theta - r_t) dt + sigma dW_t

We use exact (not Euler) discretisation for stability:
    r_{t+1} | r_t ~ N( theta + (r_t - theta) e^{-kappa dt},
                       sigma^2 (1 - e^{-2 kappa dt}) / (2 kappa) )

Calibration uses the equivalent AR(1) representation:
    r_{t+1} = a + b * r_t + eps     where b = exp(-kappa dt),
                                          a = theta * (1 - b),
                                          Var(eps) = sigma^2 (1 - b^2) / (2 kappa)
OLS on the pair (r_t, r_{t+1}) is closed-form and unbiased; ML for a Gaussian
AR(1) coincides with OLS.

The analytical zero-coupon bond price is
    P(t,T) = A(t,T) * exp(-B(t,T) r_t)
with the standard A, B functions (Brigo & Mercurio, ch. 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class VasicekParams(BaseModel):
    """Frozen parameter container."""

    model_config = ConfigDict(frozen=True)

    kappa: float = Field(gt=0, description="Mean-reversion speed (per year).")
    theta: float = Field(description="Long-run mean short rate (annualised).")
    sigma: float = Field(gt=0, description="Instantaneous volatility (per sqrt year).")
    r0: float = Field(description="Initial short rate (annualised).")


@dataclass(frozen=True)
class CalibrationResult:
    params: VasicekParams
    n_obs: int
    dt: float
    log_likelihood: float
    half_life_years: float


class VasicekModel:
    """Vasicek 1F: simulate, calibrate, price zero-coupon bonds."""

    def __init__(self, params: VasicekParams):
        self.params = params

    # ------------------------------------------------------------------ simulate
    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        dt: float,
        *,
        seed: int | None = None,
        antithetic: bool = True,
    ) -> np.ndarray:
        """Exact-discretisation MC. Returns shape (n_paths, n_steps + 1) starting at r0."""
        if n_paths <= 0 or n_steps <= 0 or dt <= 0:
            raise ValueError("n_paths, n_steps, dt must be positive.")

        rng = np.random.default_rng(seed)
        p = self.params
        b = math.exp(-p.kappa * dt)
        cond_mean_coeff = b
        cond_mean_const = p.theta * (1 - b)
        cond_var = (p.sigma ** 2) * (1 - b * b) / (2 * p.kappa)
        cond_sd = math.sqrt(cond_var)

        # Antithetic halving — generate half, mirror sign
        if antithetic and n_paths % 2 == 0:
            half = n_paths // 2
            shocks_half = rng.standard_normal((half, n_steps))
            shocks = np.vstack([shocks_half, -shocks_half])
        else:
            shocks = rng.standard_normal((n_paths, n_steps))

        r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        r[:, 0] = p.r0
        for k in range(n_steps):
            r[:, k + 1] = cond_mean_const + cond_mean_coeff * r[:, k] + cond_sd * shocks[:, k]
        return r

    # ------------------------------------------------------- analytical bond price
    def _AB(self, tau: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A(t,T), B(t,T) Vasicek functions; tau = T - t in years."""
        p = self.params
        tau = np.asarray(tau, dtype=np.float64)
        B = (1.0 - np.exp(-p.kappa * tau)) / p.kappa
        gamma = (p.theta - (p.sigma ** 2) / (2 * p.kappa ** 2)) * (B - tau)
        A_exponent = gamma - (p.sigma ** 2) * B ** 2 / (4 * p.kappa)
        A = np.exp(A_exponent)
        return A, B

    def bond_price(self, tau: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
        """Zero-coupon bond price P(t, t+tau) given short rate r at time t."""
        A, B = self._AB(tau)
        return A * np.exp(-B * np.asarray(r, dtype=np.float64))

    def zero_yield(self, tau: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
        """Continuously-compounded zero yield y(t, t+tau) = -log P / tau."""
        tau_arr = np.asarray(tau, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(tau_arr > 0, -np.log(self.bond_price(tau_arr, r)) / tau_arr, np.asarray(r))

    # ------------------------------------------------------------------- calibrate
    @classmethod
    def calibrate(cls, rates: np.ndarray, dt: float) -> CalibrationResult:
        """Calibrate to a 1D series of short-rate observations sampled at step dt.

        OLS on the AR(1) representation; analytic recovery of (kappa, theta, sigma).
        """
        rates = np.asarray(rates, dtype=np.float64).ravel()
        if rates.size < 3:
            raise ValueError("Need at least 3 observations for AR(1) calibration.")

        x = rates[:-1]
        y = rates[1:]
        n = x.size

        x_mean = x.mean()
        y_mean = y.mean()
        sxx = ((x - x_mean) ** 2).sum()
        sxy = ((x - x_mean) * (y - y_mean)).sum()
        if sxx <= 0:
            raise ValueError("Singular regressor; rates are constant.")

        b = sxy / sxx
        a = y_mean - b * x_mean
        residuals = y - (a + b * x)
        sigma_eps_sq = (residuals ** 2).sum() / (n - 2)

        # b = exp(-kappa dt) must be in (0, 1) for a stationary mean-reverting process
        if not (0 < b < 1):
            raise ValueError(
                f"Estimated AR(1) coefficient b={b:.4f} outside (0,1); the series "
                "does not appear mean-reverting at this sampling frequency."
            )

        kappa = -math.log(b) / dt
        theta = a / (1 - b)
        sigma_sq = sigma_eps_sq * (2 * kappa) / (1 - b * b)
        sigma = math.sqrt(max(sigma_sq, 1e-16))

        # Gaussian AR(1) log-likelihood (constant terms dropped)
        ll = -0.5 * n * math.log(2 * math.pi * sigma_eps_sq) - 0.5 * (residuals ** 2).sum() / sigma_eps_sq

        params = VasicekParams(kappa=kappa, theta=theta, sigma=sigma, r0=float(rates[-1]))
        return CalibrationResult(
            params=params,
            n_obs=int(rates.size),
            dt=dt,
            log_likelihood=float(ll),
            half_life_years=math.log(2) / kappa,
        )
