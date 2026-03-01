### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ cdca3dcf-8ff8-451b-8c99-1ea18607ed2a
begin
	import Pkg
	Pkg.add("OrdinaryDiffEq")
end

# ╔═╡ bd9fe7c0-1501-11f1-b999-fb0264b94c15
begin
    include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
    include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
    using .Vasicek
    using .Riccati
    using Plots
    using Printf
end

# ╔═╡ bda00ed0-1501-11f1-ac8d-95a103cce5d6
md"""
# 02 — Vasicek CPU: ODE vs Closed Form

Validate that our general Riccati ODE solver (DifferentialEquations.jl)
reproduces the Vasicek closed-form solution to numerical precision.
"""

# ╔═╡ bda00ed0-1501-11f1-b0b4-f767137982fd
md"""
## Parameters
"""

# ╔═╡ bda00ed0-1501-11f1-8af5-9b19d4926b98
begin
    p = VasicekParams(0.50, 0.025, 0.010, 0.035)  # κ, θ, σ, r0
    tenors = Float64.(1:30)  # 1–30 year maturities
    nothing
end

# ╔═╡ bda00ed0-1501-11f1-bdc5-5b670b863eb6
md"""
## Closed-form Vasicek yields
"""

# ╔═╡ bda00ed0-1501-11f1-9527-d92a93e2470c
begin
    yields_cf = yield_curve(p, tenors)
    nothing
end

# ╔═╡ bda00ed0-1501-11f1-af9d-11c971d6af11
md"""
## ODE yields (Riccati solver)
"""

# ╔═╡ bda00ed0-1501-11f1-8943-bfec4ac0989a
begin
    ap         = AffineParams(p)
    yields_ode = yield_curve_ode(ap, tenors)
    nothing
end

# ╔═╡ bda00ed0-1501-11f1-85ee-55af25716121
md"""
## Validation
"""

# ╔═╡ bda00ed0-1501-11f1-9d13-e114ae280009
begin
    result = validate_vs_vasicek(p, tenors)
    @printf "Max absolute error (ODE vs closed-form): %.2e\n" result.max_abs_error
    result
end

# ╔═╡ bda00ed0-1501-11f1-96f5-d1a694f6e663
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

# ╔═╡ bda035de-1501-11f1-89de-bbbf967bf181
begin
    errs = abs.(yields_cf .- yields_ode) .* 1e6   # in micro-percent

    plot(tenors, errs,
         xlabel="Maturity (years)",
         ylabel="|error| (μ%)",
         title="Absolute error: ODE vs closed form (micro-percent scale)",
         legend=false,
         color=:gray, lw=1.5)
end

# ╔═╡ bda035de-1501-11f1-bb39-17c9d1c7a9a2
md"""
## Inspect B(τ) and A(τ) directly
"""

# ╔═╡ bda035de-1501-11f1-a080-654db3ea9fb3
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

# ╔═╡ bda035de-1501-11f1-ad32-89855a0abdcf
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

# ╔═╡ Cell order:
# ╠═cdca3dcf-8ff8-451b-8c99-1ea18607ed2a
# ╠═bd9fe7c0-1501-11f1-b999-fb0264b94c15
# ╠═bda00ed0-1501-11f1-ac8d-95a103cce5d6
# ╠═bda00ed0-1501-11f1-b0b4-f767137982fd
# ╠═bda00ed0-1501-11f1-8af5-9b19d4926b98
# ╠═bda00ed0-1501-11f1-bdc5-5b670b863eb6
# ╠═bda00ed0-1501-11f1-9527-d92a93e2470c
# ╠═bda00ed0-1501-11f1-af9d-11c971d6af11
# ╠═bda00ed0-1501-11f1-8943-bfec4ac0989a
# ╠═bda00ed0-1501-11f1-85ee-55af25716121
# ╠═bda00ed0-1501-11f1-9d13-e114ae280009
# ╠═bda00ed0-1501-11f1-96f5-d1a694f6e663
# ╠═bda035de-1501-11f1-89de-bbbf967bf181
# ╠═bda035de-1501-11f1-bb39-17c9d1c7a9a2
# ╠═bda035de-1501-11f1-a080-654db3ea9fb3
# ╠═bda035de-1501-11f1-ad32-89855a0abdcf
