"""
Benchmark: Python — SciPy solve_ivp + JAX batched ODE
======================================================
Comparison baseline for Julia CPU/GPU benchmarks.

Run with:  python bench_python.py

Requires:  scipy, numpy, jax, jaxlib
           pip install scipy jax jaxlib   (or conda)
"""

import time
import numpy as np
from scipy.integrate import solve_ivp

# ── SciPy baseline ──────────────────────────────────────────────────────────

def riccati_rhs(tau, u, kappa, theta, sigma2):
    """Riccati ODE: u = [B, A]"""
    B, A = u
    dB = 1.0 - kappa * B
    dA = kappa * theta * B - (sigma2 / 2.0) * B**2
    return [dB, dA]

def solve_riccati_scipy(kappa, theta, sigma, r0, tenors):
    """Single Riccati ODE solve with SciPy."""
    sol = solve_ivp(
        riccati_rhs,
        [0.0, max(tenors)],
        [0.0, 0.0],
        method="RK45",
        t_eval=tenors,
        args=(kappa, theta, sigma**2),
        rtol=1e-6, atol=1e-8,
    )
    B = sol.y[0]
    A = sol.y[1]
    return (-A + B * r0) / tenors   # yields

def benchmark_scipy(N, tenors):
    """Solve N independent ODEs sequentially (Python loop)."""
    kappas = np.random.uniform(0.3, 0.7,  N).astype(np.float32)
    thetas = np.random.uniform(0.01, 0.04, N).astype(np.float32)
    sigmas = np.random.uniform(0.005, 0.02, N).astype(np.float32)
    r0s    = np.random.uniform(0.02, 0.05, N).astype(np.float32)

    t0 = time.perf_counter()
    results = [
        solve_riccati_scipy(kappas[i], thetas[i], sigmas[i], r0s[i], tenors)
        for i in range(N)
    ]
    return time.perf_counter() - t0

# ── JAX batched ODE ─────────────────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    from jax import vmap, jit
    from functools import partial
    JAX_OK = True
except ImportError:
    JAX_OK = False
    print("JAX not available — skipping JAX benchmark.")

if JAX_OK:
    @jit
    def euler_riccati_jax(kappa, theta, sigma2, r0, tenors, n_steps=500):
        """
        Fixed-step Euler integrator for Riccati ODE.
        (JAX does not ship a production-quality adaptive ODE solver;
         diffrax provides one but adds a dependency — Euler shown for fairness.)
        """
        tau_max = tenors[-1]
        dt = tau_max / n_steps
        taus = jnp.linspace(0.0, tau_max, n_steps + 1)

        def step(carry, tau):
            B, A = carry
            dB = 1.0 - kappa * B
            dA = kappa * theta * B - (sigma2 / 2.0) * B**2
            return (B + dt * dB, A + dt * dA), (B, A)

        _, (Bs, As) = jax.lax.scan(step, (0.0, 0.0), taus[:-1])
        # Interpolate at tenors
        B_at = jnp.interp(tenors, taus[1:], Bs)
        A_at = jnp.interp(tenors, taus[1:], As)
        return (-A_at + B_at * r0) / tenors

    batched_riccati = jit(vmap(
        euler_riccati_jax, in_axes=(0, 0, 0, 0, None)
    ))

    def benchmark_jax(N, tenors):
        kappas = jnp.array(np.random.uniform(0.3, 0.7,  N), dtype=jnp.float32)
        thetas = jnp.array(np.random.uniform(0.01, 0.04, N), dtype=jnp.float32)
        sigmas2= jnp.array(np.random.uniform(0.005, 0.02, N)**2, dtype=jnp.float32)
        r0s    = jnp.array(np.random.uniform(0.02, 0.05, N), dtype=jnp.float32)
        jax_tenors = jnp.array(tenors, dtype=jnp.float32)

        # warm-up (JIT compile)
        _ = batched_riccati(kappas[:10], thetas[:10], sigmas2[:10],
                            r0s[:10], jax_tenors).block_until_ready()

        t0 = time.perf_counter()
        out = batched_riccati(kappas, thetas, sigmas2,
                              r0s, jax_tenors).block_until_ready()
        return time.perf_counter() - t0

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tenors = np.arange(1, 31, dtype=np.float64)
    Ns     = [100, 500, 1_000, 2_000, 5_000, 10_000]

    print("=" * 60)
    print("Python benchmark — SciPy solve_ivp (sequential)")
    print("=" * 60)
    print(f"{'N':<8}  {'SciPy (s)':<12}")
    for N in Ns:
        t = benchmark_scipy(N, tenors)
        print(f"{N:<8}  {t:<12.4f}")

    if JAX_OK:
        print()
        print("=" * 60)
        print("JAX batched Euler (vmap + jit)")
        backend = jax.default_backend()
        print(f"Backend: {backend}")
        print("=" * 60)
        print(f"{'N':<8}  {'JAX (s)':<12}")
        jax_tenors = np.arange(1, 31, dtype=np.float32)
        for N in Ns:
            t = benchmark_jax(N, jax_tenors)
            print(f"{N:<8}  {t:<12.4f}")
