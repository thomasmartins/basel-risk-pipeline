### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ imports
begin
    include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
    include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
    using .Vasicek
    using .Riccati
    using Plots
    using Printf
end

# ╔═╡ title
md"""
# 02 — Vasicek CPU: ODE vs Closed Form

Validate that our general Riccati ODE solver (DifferentialEquations.jl)
reproduces the Vasicek closed-form solution to numerical precision.
"""

# ╔═╡ params
md"""
## Parameters
"""

# ╔═╡ param_values
begin
    p = VasicekParams(
        κ  = 0.50,   # mean reversion ~ 2 years
        θ  = 0.025,  # long-run rate   2.5 %
        σ  = 0.010,  # annual vol      1.0 %
        r0 = 0.035,  # current rate    3.5 %
    )
    tenors = Float64.(1:30)  # 1–30 year maturities
    nothing
end

# ╔═╡ closed_form
md"""
## Closed-form Vasicek yields
"""

# ╔═╡ compute_cf
begin
    yields_cf = yield_curve(p, tenors)
    nothing
end

# ╔═╡ ode_solve
md"""
## ODE yields (Riccati solver)
"""

# ╔═╡ compute_ode
begin
    ap         = AffineParams(p)
    yields_ode = yield_curve_ode(ap, tenors)
    nothing
end

# ╔═╡ validation
md"""
## Validation
"""

# ╔═╡ run_validation
begin
    result = validate_vs_vasicek(p, tenors)
    @printf "Max absolute error (ODE vs closed-form): %.2e\n" result.max_abs_error
    result
end

# ╔═╡ plot_curves
begin
    pl = plot(tenors, yields_cf .* 100,
              label="Closed form",
              lw=2, color=:steelblue,
              xlabel="Maturity (years)",
              ylabel="Yield (%)",
              title="Vasicek yield curve — ODE vs closed form",
              legend=:bottomright)

    plot!(pl, tenors, yields_ode .* 100,
          label="Riccati ODE",
          lw=2, ls=:dash, color=:firebrick)

    pl
end

# ╔═╡ plot_error
begin
    errs = abs.(yields_cf .- yields_ode) .* 1e6   # in micro-percent

    plot(tenors, errs,
         xlabel="Maturity (years)",
         ylabel="|error| (μ%)",
         title="Absolute error: ODE vs closed form (micro-percent scale)",
         legend=false,
         color=:gray, lw=1.5)
end

# ╔═╡ B_and_A
md"""
## Inspect B(τ) and A(τ) directly
"""

# ╔═╡ plot_BA
begin
    Bvals = B_vasicek.(Ref(p), tenors)
    Avals = A_vasicek.(Ref(p), tenors)

    p1 = plot(tenors, Bvals,
              title="B(τ) — loading on r₀",
              xlabel="τ (years)", ylabel="B",
              label="B(τ)", color=:steelblue, lw=2)

    p2 = plot(tenors, Avals,
              title="A(τ) — intercept",
              xlabel="τ (years)", ylabel="A",
              label="A(τ)", color=:darkorange, lw=2)

    plot(p1, p2, layout=(1,2), size=(800, 300))
end

# ╔═╡ interpretation
md"""
## Interpretation

- **B(τ)** saturates at $1/\kappa$ as $\tau\to\infty$ → long yields are
  less sensitive to the current short rate than short yields.
- **A(τ)** encodes the risk-neutral drift and convexity adjustment.
  It is always negative (bond prices are pulled down by Jensen's inequality).
- The ODE error is $\mathcal{O}(10^{-9})$ — far below any financial precision
  requirement.  The ODE formulation is therefore a valid substitute for the
  closed form and generalises to any affine model.

→ Next: **`03_gpu_scaling.jl`** — why GPU matters only at scale.
"""
