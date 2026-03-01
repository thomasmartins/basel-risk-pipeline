### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ imports
begin
    include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
    include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
    include(joinpath(@__DIR__, "..", "src", "gpu_ensemble.jl"))
    include(joinpath(@__DIR__, "..", "src", "bayes.jl"))
    using .Vasicek
    using .Riccati
    using .GPUEnsemble
    using .BayesCalibration
    using Plots
    using Statistics
    using Printf
end

# ╔═╡ title
md"""
# 04 — Bayesian Calibration: Credible Bands for NMD Rates

We treat Vasicek parameters as uncertain (drawn from a posterior)
and propagate that uncertainty into the NMD deposit rate via the
affine term structure model.

Full pipeline:
$$p(\kappa,\theta,\sigma,r_0) \;\xrightarrow{\text{GPU ODE}}\; y(\tau) \;\xrightarrow{\text{pass-through}}\; d(\tau) = \alpha + \beta\,y(\tau)$$
"""

# ╔═╡ posterior_setup
md"""
## Mock posterior

In a production system, these distributions would be the output of
MCMC (e.g. Turing.jl) run on a panel of observed deposit / market rates.
Here we use calibrated Normal distributions consistent with Euro-area data.
"""

# ╔═╡ build_posterior
begin
    post = MockPosterior(
        κ_mean=0.50f0,   κ_std=0.12f0,
        θ_mean=0.025f0,  θ_std=0.006f0,
        σ_mean=0.010f0,  σ_std=0.002f0,
        r0_mean=0.035f0, r0_std=0.004f0,
    )
    nothing
end

# ╔═╡ draw_samples
begin
    N      = 5_000
    draws  = sample_draws(post, N)
    tenors = Float32.(1:30)
    @printf "Drew %d parameter samples from mock posterior.\n" N
    nothing
end

# ╔═╡ gpu_solve
md"""
## GPU ensemble solve → yield credible bands
"""

# ╔═╡ compute_bands
begin
    bands = credible_bands(draws, tenors; use_gpu=true, level=0.90)
    @printf "Yield credible bands computed. Median 10Y yield: %.2f%%\n" bands.median[10]*100
    nothing
end

# ╔═╡ plot_yield_bands
begin
    pl = plot(bands.tenors, bands.median .* 100,
              ribbon=(
                  (bands.median .- bands.lower) .* 100,
                  (bands.upper  .- bands.median) .* 100
              ),
              fillalpha=0.25,
              label="Median yield + 90% credible band",
              color=:steelblue,
              lw=2,
              xlabel="Maturity (years)",
              ylabel="Yield (%)",
              title="Vasicek yield curve — Bayesian credible bands\n(N=$N posterior draws, GPU ensemble)")

    pl
end

# ╔═╡ nmd_section
md"""
## NMD deposit rate credible bands

Apply affine pass-through:  $d(\tau) = \alpha + \beta \cdot y(\tau)$

with $\alpha = 20\,\text{bps}$ (bank margin) and $\beta = 0.60$ (60% pass-through).
"""

# ╔═╡ compute_nmd
begin
    nmd = nmd_rate_bands(bands; α_pass=0.002, β_pass=0.60)
    @printf "NMD rate credible bands computed. Median 10Y NMD rate: %.2f%%\n" nmd.median[10]*100
    nothing
end

# ╔═╡ plot_nmd_bands
begin
    pl_nmd = plot(nmd.tenors, nmd.median .* 100,
                  ribbon=(
                      (nmd.median .- nmd.lower) .* 100,
                      (nmd.upper  .- nmd.median) .* 100
                  ),
                  fillalpha=0.25,
                  label="Median NMD rate + 90% credible band",
                  color=:darkorange,
                  lw=2,
                  xlabel="Reference maturity (years)",
                  ylabel="NMD deposit rate (%)",
                  title="NMD rate — Bayesian uncertainty propagation\n(α=20bps, β=0.60)")

    pl_nmd
end

# ╔═╡ joint_plot
begin
    p1 = plot(bands.tenors, bands.median .* 100,
              ribbon=(
                  (bands.median .- bands.lower) .* 100,
                  (bands.upper  .- bands.median) .* 100
              ),
              fillalpha=0.2, color=:steelblue, lw=2,
              label="90% band", title="Model yield curve",
              xlabel="Maturity (yr)", ylabel="Rate (%)", legend=:bottomright)

    p2 = plot(nmd.tenors, nmd.median .* 100,
              ribbon=(
                  (nmd.median .- nmd.lower) .* 100,
                  (nmd.upper  .- nmd.median) .* 100
              ),
              fillalpha=0.2, color=:darkorange, lw=2,
              label="90% band", title="NMD deposit rate",
              xlabel="Reference maturity (yr)", ylabel="Rate (%)", legend=:bottomright)

    plot(p1, p2, layout=(1,2), size=(900, 350),
         suptitle="Bayesian Uncertainty Propagation: Vasicek → NMD Rate")
end

# ╔═╡ summary
md"""
## Summary

| Component | Detail |
|-----------|--------|
| Model | Vasicek affine term structure |
| ODE solver | Riccati system, Tsit5, reltol=1e-6 |
| Posterior draws | $N = 5{,}000$ (mock; replace with MCMC) |
| Batch compute | DiffEqGPU.jl EnsembleGPUArray |
| Pass-through | $d(\tau) = 0.002 + 0.60 \cdot y(\tau)$ |
| Output | Pointwise median + 90% credible band for $d(\tau)$ |

**Key insight:** the credible band on the NMD rate is entirely driven by
uncertainty in macroeconomic parameters $(\kappa, \theta, \sigma, r_0)$.
In practice, $\beta$ is also uncertain — adding it as a parameter widens
the band further and is straightforward to include.
"""
