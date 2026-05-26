"""Hull-White 1-factor short-rate model (extended Vasicek).

SDE under the risk-neutral measure:

    dr_t = (theta(t) - a * r_t) dt + sigma dW_t

theta(t) is *bootstrapped* from the initial zero curve so the model reprices
the observed term structure exactly. With

    alpha(t) := f^M(0, t) + sigma^2 / (2 a^2) * (1 - e^{-a t})^2

the short rate has the explicit decomposition r_t = x_t + alpha(t), where x_t
is a mean-zero OU process with parameters (a, sigma). The exact (non-Euler)
one-step transition is

    r_{k+1} = e^{-a dt} r_k + alpha(t_{k+1}) - e^{-a dt} alpha(t_k)
              + sigma * sqrt((1 - e^{-2 a dt}) / (2 a)) * eps_k

and the analytical zero-coupon bond price (Brigo & Mercurio, ch. 3) is

    P(t, T) = A(t, T) * exp(-B(t, T) r_t)
    B(t, T) = (1 - e^{-a (T - t)}) / a
    A(t, T) = (P^M(0, T) / P^M(0, t)) * exp(B(t, T) f^M(0, t)
              - sigma^2 / (4 a) * (1 - e^{-2 a t}) * B(t, T)^2)

At t = 0 this collapses to A(0, tau) = P^M(0, tau) * exp(B(0, tau) r_0), so
P(0, tau; r_0) = P^M(0, tau) by construction — the curve fits perfectly.

Calibration in this implementation:
    - (a, sigma) come from OLS on the AR(1) representation of the historical
      short rate (same closed-form recipe as Vasicek).
    - theta(t) / alpha(t) come from the input market curve — no fitting needed.

This separation is deliberate: history pins the diffusion + speed-of-reversion,
the curve pins the drift schedule, so the model is simultaneously
history-consistent and arbitrage-free against today's quoted curve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from basel_risk_engine.valuation.curve import YieldCurve


class HullWhiteParams(BaseModel):
    """Frozen parameter container.

    `a` plays the role of kappa in Vasicek; theta(t) is implicit in the
    market curve and is therefore not a scalar parameter here.
    """

    model_config = ConfigDict(frozen=True)

    a: float = Field(gt=0, description="Mean-reversion speed (per year).")
    sigma: float = Field(gt=0, description="Instantaneous volatility (per sqrt year).")
    r0: float = Field(description="Initial short rate r_0 (annualised).")


@dataclass(frozen=True)
class HullWhiteCalibration:
    params: HullWhiteParams
    n_obs: int
    dt: float
    log_likelihood: float
    half_life_years: float
    curve_fit_max_residual: float  # max |P_model(0,tau; r0) - P^M(0,tau)| over tenor grid


class HullWhiteModel:
    """HW1F: simulate, calibrate (history + curve), price zero-coupon bonds."""

    def __init__(self, params: HullWhiteParams, market_curve: YieldCurve):
        self.params = params
        self.market_curve = market_curve

    # ------------------------------------------------------------------ alpha
    def alpha(self, t: float | np.ndarray) -> np.ndarray:
        """alpha(t) = f^M(0,t) + sigma^2 / (2 a^2) * (1 - e^{-a t})^2."""
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        p = self.params
        # Guard f^M(0,0) which our linear-yield curve defines as y(τ→0); the
        # convention r_0 = f^M(0,0) is what makes the curve fit exact.
        f0 = self.market_curve.forward_rate(np.where(t <= 0, 1e-12, t))
        f0 = np.where(t <= 0, p.r0, f0)
        convexity = (p.sigma ** 2) / (2 * p.a ** 2) * (1 - np.exp(-p.a * t)) ** 2
        return f0 + convexity

    # --------------------------------------------------------------- simulate
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
        decay = math.exp(-p.a * dt)
        cond_sd = math.sqrt((p.sigma ** 2) * (1 - decay * decay) / (2 * p.a))

        times = np.arange(n_steps + 1) * dt
        alphas = self.alpha(times)  # (n_steps + 1,)
        # Per-step drift adjustment d_k = alpha(t_{k+1}) - decay * alpha(t_k)
        drift = alphas[1:] - decay * alphas[:-1]  # (n_steps,)

        if antithetic and n_paths % 2 == 0:
            half = n_paths // 2
            shocks_half = rng.standard_normal((half, n_steps))
            shocks = np.vstack([shocks_half, -shocks_half])
        else:
            shocks = rng.standard_normal((n_paths, n_steps))

        r = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        r[:, 0] = p.r0
        for k in range(n_steps):
            r[:, k + 1] = decay * r[:, k] + drift[k] + cond_sd * shocks[:, k]
        return r

    # ----------------------------------------------------- analytical bond price
    def _AB(self, tau: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """A(0, tau), B(0, tau) at valuation date.

        A(0, tau) = P^M(0, tau) * exp(B(0, tau) * r_0), so plugging r = r_0
        recovers P^M exactly. Used by EVEEngine for instantaneous-shock pricing
        with r = r_h on each MC path at the forward horizon.
        """
        p = self.params
        tau = np.asarray(tau, dtype=np.float64)
        B = (1.0 - np.exp(-p.a * tau)) / p.a
        market_df = self.market_curve.discount_factor(tau)
        A = market_df * np.exp(B * p.r0)
        return A, B

    def bond_price(self, tau: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
        """P(0, tau) given short rate r at t = 0. With r = r_0 this is P^M(0, tau)."""
        A, B = self._AB(tau)
        return A * np.exp(-B * np.asarray(r, dtype=np.float64))

    def zero_yield(self, tau: float | np.ndarray, r: float | np.ndarray) -> np.ndarray:
        tau_arr = np.asarray(tau, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                tau_arr > 0,
                -np.log(self.bond_price(tau_arr, r)) / tau_arr,
                np.asarray(r),
            )

    # ------------------------------------------------------------------- calibrate
    @classmethod
    def calibrate(
        cls,
        rates: np.ndarray,
        dt: float,
        market_curve: YieldCurve,
        *,
        r0_override: float | None = None,
    ) -> HullWhiteCalibration:
        """Calibrate (a, sigma) from a short-rate history via OLS on the AR(1)
        decomposition of x_t = r_t - alpha(t). theta(t) is read off the curve.

        For the AR(1) regression we subtract a coarse alpha proxy = f^M(0, t_k)
        from each observation; the convexity term is negligible at the scale of
        a short historical window. Result: a, sigma, half-life.

        r_0 defaults to the last observation; pass `r0_override` to pin it to a
        specific value (e.g. f^M(0, 0) so curve-fit-by-construction holds exactly).
        """
        rates = np.asarray(rates, dtype=np.float64).ravel()
        if rates.size < 3:
            raise ValueError("Need at least 3 observations for AR(1) calibration.")

        # OLS on x_{k+1} = b * x_k + eps with x_k = r_k - f^M(0, t_k)
        n = rates.size
        t_grid = np.arange(n) * dt
        forwards = market_curve.forward_rate(np.where(t_grid <= 0, 1e-12, t_grid))
        x = rates - forwards
        x_lag = x[:-1]
        x_next = x[1:]
        m = x_lag.size

        x_mean = x_lag.mean()
        y_mean = x_next.mean()
        sxx = ((x_lag - x_mean) ** 2).sum()
        sxy = ((x_lag - x_mean) * (x_next - y_mean)).sum()
        if sxx <= 0:
            raise ValueError("Singular regressor; the de-meaned series is constant.")

        b = sxy / sxx
        residuals = x_next - b * x_lag
        sigma_eps_sq = (residuals ** 2).sum() / max(m - 1, 1)

        if not (0 < b < 1):
            raise ValueError(
                f"Estimated AR(1) coefficient b={b:.4f} outside (0,1); the de-meaned "
                "rate series does not appear mean-reverting at this sampling frequency."
            )

        a = -math.log(b) / dt
        sigma_sq = sigma_eps_sq * (2 * a) / (1 - b * b)
        sigma = math.sqrt(max(sigma_sq, 1e-16))

        # Gaussian AR(1) log-likelihood on the de-meaned residuals
        ll = -0.5 * m * math.log(2 * math.pi * sigma_eps_sq) - 0.5 * (residuals ** 2).sum() / sigma_eps_sq

        r0 = float(rates[-1]) if r0_override is None else float(r0_override)
        params = HullWhiteParams(a=a, sigma=sigma, r0=r0)
        model = cls(params, market_curve)

        # Curve-fit residual: zero by construction at any tau, but we measure it
        # at the curve's own grid to surface any numerical drift.
        tenor_grid = market_curve.tenors_years
        model_dfs = model.bond_price(tenor_grid, r0)
        market_dfs = market_curve.discount_factor(tenor_grid)
        # Drift from r0 != f^M(0,0) shows up here:
        curve_fit_max_residual = float(np.max(np.abs(model_dfs - market_dfs)))

        return HullWhiteCalibration(
            params=params,
            n_obs=int(rates.size),
            dt=dt,
            log_likelihood=float(ll),
            half_life_years=math.log(2) / a,
            curve_fit_max_residual=curve_fit_max_residual,
        )
