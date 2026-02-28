"""
    Vasicek closed-form affine term structure
    ==========================================

    Under the risk-neutral measure Q, the short rate follows:

        dr(t) = κ(θ - r(t)) dt + σ dW(t)

    Bond price for maturity τ = T - t:

        P(t, T) = exp( A(τ) - B(τ) r(t) )

    where A and B satisfy the Riccati ODEs (see riccati.jl).
    For Vasicek, these have the closed-form solution below.
"""

module Vasicek

export VasicekParams, B_vasicek, A_vasicek, bond_price, yield_curve

"""
Parameters of the Vasicek model.

  κ  : mean-reversion speed  (κ > 0)
  θ  : long-run mean of r
  σ  : volatility
  r0 : current short rate
"""
struct VasicekParams
    κ::Float64
    θ::Float64
    σ::Float64
    r0::Float64
end

"""
    B_vasicek(p, τ)

ODE factor B(τ) — the loading on r(t) in the log bond price.

Closed form:  B(τ) = (1 - exp(-κτ)) / κ
"""
function B_vasicek(p::VasicekParams, τ::Real)
    (; κ) = p
    return (1.0 - exp(-κ * τ)) / κ
end

"""
    A_vasicek(p, τ)

Intercept A(τ) in the log bond price.

Let  θ̄ = θ - σ²/(2κ²)   (risk-adjusted long-run mean)

Closed form:
    A(τ) = (B(τ) - τ) * θ̄  -  σ² B(τ)² / 4κ
"""
function A_vasicek(p::VasicekParams, τ::Real)
    (; κ, θ, σ) = p
    B = B_vasicek(p, τ)
    θ_bar = θ - σ^2 / (2κ^2)
    return (B - τ) * θ_bar - σ^2 * B^2 / (4κ)
end

"""
    bond_price(p, τ)

Zero-coupon bond price P(0, τ) = exp(A(τ) - B(τ) r₀).
"""
function bond_price(p::VasicekParams, τ::Real)
    return exp(A_vasicek(p, τ) - B_vasicek(p, τ) * p.r0)
end

"""
    yield_curve(p, tenors)

Continuously-compounded zero yields y(τ) = -log P(0,τ) / τ
over a vector of maturities `tenors` (in years).

Returns a vector of yields (annualised, in the same units as κ, θ, σ).
"""
function yield_curve(p::VasicekParams, tenors::AbstractVector)
    return [-log(bond_price(p, τ)) / τ for τ in tenors]
end

end # module
