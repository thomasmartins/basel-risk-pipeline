"""
    General affine term structure via Riccati ODEs
    ================================================

    For any single-factor affine model where the short rate and its
    variance are affine in the state variable x(t):

        r(t)   = δ₀ + δ₁ x(t)
        Var(dx) = α₀ + α₁ x(t)        (Vasicek: α₀=σ², α₁=0)
        E[dx]   = κ(θ - x(t)) dt

    the bond price is still  P = exp(A(τ) - B(τ) x(t)), but A and B
    now solve the Riccati ODE system (Duffie & Kan 1996):

        dB/dτ = δ₁ - κ B - (α₁/2) B²        [Riccati in B]
        dA/dτ = -δ₀ + κθ B - (α₀/2) B²      [linear in A once B known]

    with boundary conditions  A(0) = 0,  B(0) = 0.

    Vasicek is the special case:  δ₀=0, δ₁=1, α₀=σ², α₁=0.

    We solve this with OrdinaryDiffEq.jl and compare against the
    Vasicek closed form as a validation.
"""

module Riccati

using OrdinaryDiffEq
using ..Vasicek: VasicekParams, yield_curve as vasicek_yield_curve

export AffineParams, solve_riccati, yield_curve_ode, validate_vs_vasicek

"""
Parameters for the general single-factor affine model.

  κ, θ : mean-reversion speed and level
  δ₀,δ₁: affine mapping r = δ₀ + δ₁ x
  α₀,α₁: variance specification  Var(dx) = α₀ + α₁ x
"""
struct AffineParams
    κ::Float64
    θ::Float64
    δ₀::Float64
    δ₁::Float64
    α₀::Float64
    α₁::Float64
    x0::Float64   # current value of state variable x(t)
end

"""
    AffineParams(p::VasicekParams)

Convenience constructor: embed a Vasicek model into the general affine framework.
"""
function AffineParams(p::VasicekParams)
    return AffineParams(p.κ, p.θ, 0.0, 1.0, p.σ^2, 0.0, p.r0)
end

"""
    riccati_ode!(du, u, params, τ)

In-place ODE right-hand side.

State vector  u = [B, A].

  dB/dτ = δ₁ - κ B - (α₁/2) B²
  dA/dτ = -δ₀ + κθ B - (α₀/2) B²
"""
function riccati_ode!(du, u, params, τ)
    B, A = u
    κ, θ, δ₀, δ₁, α₀, α₁ = params
    du[1] = δ₁ - κ * B - (α₁ / 2) * B^2
    du[2] = -δ₀ - κ * θ * B + (α₀ / 2) * B^2
end

"""
    solve_riccati(p::AffineParams, τ_max; τ_save=nothing)

Solve the Riccati system from τ=0 to τ=τ_max.

Returns a DiffEq solution object. If `τ_save` is provided (a vector
of maturities), the solution is saved at those points only.
"""
function solve_riccati(p::AffineParams, τ_max::Real; τ_save=nothing)
    u0     = [0.0, 0.0]           # B(0) = A(0) = 0
    params = (p.κ, p.θ, p.δ₀, p.δ₁, p.α₀, p.α₁)
    tspan  = (0.0, τ_max)

    prob = ODEProblem(riccati_ode!, u0, tspan, params)

    if isnothing(τ_save)
        return solve(prob, Tsit5(); reltol=1e-10, abstol=1e-12)
    else
        return solve(prob, Tsit5(); saveat=τ_save, reltol=1e-10, abstol=1e-12)
    end
end

"""
    yield_curve_ode(p::AffineParams, tenors)

Yield curve via ODE: y(τ) = (-A(τ) + B(τ) x₀) / τ.
"""
function yield_curve_ode(p::AffineParams, tenors::AbstractVector)
    τ_max = maximum(tenors)
    sol   = solve_riccati(p, τ_max; τ_save=tenors)

    yields = Vector{Float64}(undef, length(tenors))
    for (i, τ) in enumerate(tenors)
        B, A = sol(τ)
        yields[i] = (-A + B * p.x0) / τ
    end
    return yields
end

"""
    validate_vs_vasicek(p::VasicekParams, tenors; tol=1e-8)

Compare ODE yields against Vasicek closed-form yields.
Returns (closed_form, ode, max_abs_error).
Throws an AssertionError if max error exceeds `tol`.
"""
function validate_vs_vasicek(p::VasicekParams, tenors::AbstractVector; tol=1e-8)
    yv  = vasicek_yield_curve(p, tenors)
    ap  = AffineParams(p)
    yod = yield_curve_ode(ap, tenors)
    err = maximum(abs.(yv .- yod))
    @assert err < tol "Validation failed: max error $err >= tol $tol"
    return (closed_form=yv, ode=yod, max_abs_error=err)
end

end # module
