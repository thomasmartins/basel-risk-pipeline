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
    include(joinpath(@__DIR__, "..", "src", "nmd_segments.jl"))
    using .Vasicek
    using .Riccati
    using .NMDSegments
    using Plots
    using Printf
end

# ╔═╡ title
md"""
# 05 — The NMD (B, D, V) ODE: Balance, Discount, Value

This notebook derives and solves the ODE that converts a static yield
curve into the **economic value of a deposit segment** (EVE contribution).

The Riccati solver from earlier gives us the yield curve y(τ).
This notebook picks up from there and values the deposit franchise.
"""

# ╔═╡ deriv
md"""
## Derivation

Fix a deposit segment $s$.  Under a **static curve** (rates frozen at $t=0$):

| Quantity | Formula |
|----------|---------|
| Deposit rate | $r_{\text{dep}} = \alpha_s + \beta_s \cdot y(\tau_s^*)$ |
| Run-off intensity | $\lambda_s = \lambda_{0,s} \cdot e^{\gamma_s [y(\tau_s^{\text{alt}}) - r_{\text{dep}}]}$ |
| Balance at age $u$ | $B(u) = B_0 \cdot e^{-\lambda_s u}$ |
| Spread at age $u$ | $\text{spr} = r - r_{\text{dep}}$ (constant) |

Economic value = PV of net interest margin earned on remaining balance:
$$V_s(0) = \int_0^U e^{-ru} \cdot B(u) \cdot \text{spr} \; du$$

Introduce state variables $D(u) = e^{-ru}$ and $V_{\text{acc}}(u) = \int_0^u D B \cdot \text{spr} \; dv$:
$$\frac{d}{du}\begin{bmatrix}B \\ D \\ V_{\text{acc}}\end{bmatrix} =
\begin{bmatrix}-\lambda B \\ -r D \\ D \cdot B \cdot \text{spr}\end{bmatrix},
\quad \begin{bmatrix}B(0)\\D(0)\\V(0)\end{bmatrix} = \begin{bmatrix}B_0\\1\\0\end{bmatrix}$$

$V_s(0) = V_{\text{acc}}(U)$ at the integration horizon $U$ (e.g. 30–60 years).

**Closed form** (static curve, horizon $U$):
$$V_s(0) = \frac{B_0 \cdot \text{spr}}{r + \lambda_s}\left[1 - e^{-(r+\lambda_s)U}\right]$$
"""

# ╔═╡ setup_curve
md"""
## Setup: Vasicek curve + single segment
"""

# ╔═╡ vasicek_params
begin
    vp     = VasicekParams(0.50, 0.025, 0.010, 0.035)
    tenors = Float32.(1:30)
    yields = Float32.(Vasicek.yield_curve(vp, Float64.(tenors)))
    scen   = CurveScenario(Float32(vp.r0), yields, tenors, "base")
    nothing
end

# ╔═╡ segment_def
begin
    seg = SegmentParams(
        "Retail savings",
        -0.003f0,   # α: bank pays 30bp below anchor
         0.40f0,    # β: 40% pass-through (sticky)
         2.0f0,     # τ*: anchored to 2Y rate
         5.0f0,     # τ_alt: customers compare to 5Y
         0.10f0,    # λ₀: 10%/yr baseline run-off
         2.0f0,     # γ: moderate rate sensitivity
         1.0f0,     # B₀: normalised to 1
    )
    @printf "Deposit rate:          %.4f%%\n" deposit_rate(seg, scen)*100
    @printf "Withdrawal intensity:  %.4f /yr\n" withdrawal_intensity(seg, scen)
    @printf "Spread (r - r_dep):    %.4f%%\n" (scen.r0 - deposit_rate(seg, scen))*100
    nothing
end

# ╔═╡ validate
md"""
## Validation: ODE vs closed form
"""

# ╔═╡ run_validate
begin
    U = 60f0
    eve_cf  = eve_static_closedform(seg, scen; U=U)
    eve_ode = solve_bv_cpu([seg], scen; U=U)[1]

    @printf "Closed form EVE:  %.6f\n" eve_cf
    @printf "ODE EVE:          %.6f\n" eve_ode
    @printf "Absolute error:   %.2e\n" abs(eve_cf - eve_ode)
    nothing
end

# ╔═╡ profile_plots
md"""
## Balance, discount, and spread integrand profiles
"""

# ╔═╡ compute_profiles
begin
    us      = range(0f0, U, length=300)
    λ_val   = withdrawal_intensity(seg, scen)
    r_val   = scen.r0
    spr_val = r_val - deposit_rate(seg, scen)

    B_prof = seg.B₀ .* exp.(-λ_val .* us)
    D_prof = exp.(-r_val .* us)
    intgd  = D_prof .* B_prof .* spr_val   # integrand of EVE

    p1 = plot(us, B_prof,
              label="B(u): balance", color=:steelblue, lw=2,
              xlabel="Age u (years)", ylabel="Balance (normalised)",
              title="Balance decay",  legend=:topright)

    p2 = plot(us, intgd .* 100,
              label="D(u)·B(u)·spread × 100", color=:darkorange, lw=2,
              xlabel="Age u (years)", ylabel="Discounted spread (bps·balance)",
              title="Integrand of EVE", legend=:topright)
    hline!(p2, [0], ls=:dash, color=:black, lw=1, label="")

    plot(p1, p2, layout=(1,2), size=(850, 320))
end

# ╔═╡ beta_sensitivity
md"""
## EVE vs pass-through β

When β = 1 the deposit rate perfectly tracks the anchor yield → spread is
determined only by α, and EVE reflects the margin on a decaying balance.
When β = 0 the deposit rate is fixed → large spread in high-rate environments
but also large withdrawal risk (customers leave).
"""

# ╔═╡ plot_beta
begin
    betas   = range(0f0, 1f0, length=50)
    eves_β  = Float32[]
    for β in betas
        s_β = SegmentParams(seg.name, seg.α, Float32(β), seg.τ_anchor,
                            seg.τ_alt, seg.λ₀, seg.γ, seg.B₀)
        push!(eves_β, eve_static_closedform(s_β, scen))
    end

    plot(betas, eves_β,
         xlabel="Pass-through β", ylabel="EVE",
         title="EVE as a function of pass-through β\n(r₀=3.5%, α=-30bps, λ₀=10%/yr)",
         lw=2, color=:steelblue, legend=false)
    vline!([seg.β], ls=:dash, color=:firebrick, label="β = $(seg.β)")
end

# ╔═╡ gamma_sensitivity
md"""
## EVE vs withdrawal sensitivity γ

Higher γ means customers are more rate-sensitive: they leave faster when
market rates exceed the deposit rate. This shortens the effective duration
of the NMD book and reduces (or increases) EVE depending on the sign of spread.
"""

# ╔═╡ plot_gamma
begin
    gammas  = range(0f0, 5f0, length=60)
    eves_γ  = Float32[]
    lambdas = Float32[]
    for γ in gammas
        s_γ = SegmentParams(seg.name, seg.α, seg.β, seg.τ_anchor,
                            seg.τ_alt, seg.λ₀, Float32(γ), seg.B₀)
        push!(eves_γ,  eve_static_closedform(s_γ, scen))
        push!(lambdas, withdrawal_intensity(s_γ, scen))
    end

    pg = plot(gammas, eves_γ,
              xlabel="Withdrawal sensitivity γ", ylabel="EVE",
              title="EVE vs γ  (r₀=3.5%, β=0.40, λ₀=10%/yr)",
              lw=2, color=:darkorange, legend=false)
    vline!(pg, [seg.γ], ls=:dash, color=:firebrick, label="γ = $(seg.γ)")

    pl = plot(gammas, lambdas,
              xlabel="γ", ylabel="λ (run-off rate)",
              title="Run-off intensity λ vs γ",
              lw=2, color=:gray, legend=false)

    plot(pg, pl, layout=(1,2), size=(850, 320))
end

# ╔═╡ summary
md"""
## Key takeaways

| Parameter | Effect on EVE |
|-----------|--------------|
| **β (pass-through)** | Higher β → spread driven by α only; less sensitivity to rate level |
| **γ (withdrawal sensitivity)** | Higher γ → faster run-off when rates are high → lower effective duration |
| **λ₀ (baseline run-off)** | Higher λ₀ → shorter effective maturity → lower EVE |
| **α (margin)** | Pure shift in EVE — no rate sensitivity |

→ Next: **`06_nmd_eve_shocks.jl`** — multi-segment portfolio + IRRBB shock table.
"""
