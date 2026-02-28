"""
    NMD Segments: deposit rate rule, withdrawal intensity, (B,D,V) ODE
    ====================================================================

    This module implements the NMD-specific layer on top of the risk-free
    curve computed by the Riccati solver (riccati.jl).

    For each deposit segment s the model has three pieces:

    1. DEPOSIT RATE RULE
       r_dep^s(t) = α_s + β_s · y(t, τ_s*)
       α_s  : bank margin (constant spread, e.g. -30 bps)
       β_s  : pass-through ∈ [0,1]  (1 = full, 0 = completely sticky)
       τ_s* : anchor maturity (e.g. 3M, 1Y, 2Y)

    2. WITHDRAWAL INTENSITY
       λ_s(t) = λ₀_s · exp(γ_s · [y(t, τ_s^alt) - r_dep^s(t)])
       λ₀_s : baseline run-off rate (per year, e.g. 0.10 = 10%/yr)
       γ_s  : rate sensitivity (higher γ → deposits leave faster when
              market pays more than the bank)
       τ_s^alt: alternative instrument maturity customers compare against

    3. BALANCE / DISCOUNT / VALUE ODE
       Under a *static* curve (rates frozen at current values):

         dB/du = -λ_s · B                   [balance decay]
         dD/du = -r · D                      [discount factor, D(0)=1]
         dV/du =  D · B · spread             [accumulated spread PV]

       where spread = r - r_dep^s (net interest margin on remaining balance).

       Integrating from u=0 to U gives:
         B(U)  → balance remaining at horizon
         D(U)  → discount factor at horizon
         V(U)  → EVE contribution of segment s   ← the key output

    CLOSED FORM (validation, static curve only):
       V(0) = B₀ · spread / (r + λ) · [1 - exp(-(r+λ)·U)]
       → B₀ · spread / (r + λ)  as U → ∞

    GPU: the ODE is 3D (tiny) but we solve it for many (segment, scenario,
    posterior draw) combinations simultaneously via EnsembleGPUArray.
"""

module NMDSegments

using OrdinaryDiffEq
using DiffEqGPU
using StaticArrays
using CUDA

export SegmentParams, CurveScenario
export yield_at, deposit_rate, withdrawal_intensity
export eve_static_closedform
export solve_bv_cpu, solve_bv_gpu

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

"""
Parameters for one NMD segment (retail, savings, SME, corporate, …).
All rate quantities are in annual decimal (0.025 = 2.5%).
"""
struct SegmentParams
    name::String
    α::Float32          # constant spread / bank margin
    β::Float32          # pass-through coefficient ∈ [0,1]
    τ_anchor::Float32   # anchor maturity for deposit rate (years)
    τ_alt::Float32      # alternative maturity for withdrawal intensity
    λ₀::Float32         # baseline run-off rate (per year)
    γ::Float32          # sensitivity of run-off to rate gap
    B₀::Float32         # initial balance (normalised; e.g. 1.0 = 100%)
end

"""
A pre-computed yield curve scenario (static or shocked).
`yields[i]` is the continuously-compounded zero yield at `tenors[i]` years.
`r0` is the current short rate (= y(0⁺)).
"""
struct CurveScenario
    r0::Float32
    yields::Vector{Float32}
    tenors::Vector{Float32}
    label::String
end

# ---------------------------------------------------------------------------
# Curve interpolation
# ---------------------------------------------------------------------------

"""
    yield_at(scen, τ)

Linearly interpolate the yield curve at maturity τ (years).
Clamps to the first/last grid point outside the range.
"""
function yield_at(scen::CurveScenario, τ::Float32)
    (; yields, tenors) = scen
    τ = clamp(τ, tenors[1], tenors[end])
    idx = searchsortedlast(tenors, τ)
    idx = clamp(idx, 1, length(tenors) - 1)
    τ₁, τ₂ = tenors[idx], tenors[idx + 1]
    y₁, y₂ = yields[idx], yields[idx + 1]
    return y₁ + (y₂ - y₁) * (τ - τ₁) / (τ₂ - τ₁)
end

# ---------------------------------------------------------------------------
# Rate and intensity
# ---------------------------------------------------------------------------

"""
    deposit_rate(seg, scen)

Deposit rate for segment `seg` under curve scenario `scen`:
    r_dep = α + β · y(τ_anchor)
"""
function deposit_rate(seg::SegmentParams, scen::CurveScenario)
    return seg.α + seg.β * yield_at(scen, seg.τ_anchor)
end

"""
    withdrawal_intensity(seg, scen)

Run-off intensity:
    λ = λ₀ · exp(γ · [y(τ_alt) - r_dep])
Higher rate gap → faster outflow.
"""
function withdrawal_intensity(seg::SegmentParams, scen::CurveScenario)
    r_dep = deposit_rate(seg, scen)
    y_alt = yield_at(scen, seg.τ_alt)
    return seg.λ₀ * exp(seg.γ * (y_alt - r_dep))
end

# ---------------------------------------------------------------------------
# Closed form (static curve, infinite horizon limit)
# ---------------------------------------------------------------------------

"""
    eve_static_closedform(seg, scen; U=60f0)

Analytic EVE for a static curve truncated at horizon U years:
    V(0) = B₀ · spread / (r + λ) · [1 - exp(-(r+λ)·U)]

Useful for validating the ODE solver.
"""
function eve_static_closedform(seg::SegmentParams, scen::CurveScenario;
                                U::Float32=60f0)
    r      = scen.r0
    λ      = withdrawal_intensity(seg, scen)
    spread = r - deposit_rate(seg, scen)
    denom  = r + λ
    if abs(denom) < 1f-8
        return seg.B₀ * spread * U
    end
    return seg.B₀ * spread / denom * (1f0 - exp(-denom * U))
end

# ---------------------------------------------------------------------------
# ODE system
# ---------------------------------------------------------------------------

# State:   u = [B, D, V_acc]
# Params:  p = (λ, r, spread)   all scalars, all constant for static curve
#
#   dB/du = -λ B
#   dD/du = -r D
#   dV/du =  D · B · spread

"""In-place ODE for CPU solvers."""
function bv_ode!(du, u, p, s)
    B, D, V_acc = u
    λ, r, spread = p
    du[1] = -λ * B
    du[2] = -r * D
    du[3] =  D * B * spread
end

"""Out-of-place ODE with StaticArrays (required for GPU kernels)."""
function bv_ode_sa(u::SVector{3,T}, p, s) where T
    B, D, V_acc = u
    λ = T(p[1]); r = T(p[2]); spread = T(p[3])
    return SVector{3,T}(-λ * B, -r * D, D * B * spread)
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function _bv_params(seg::SegmentParams, scen::CurveScenario)
    λ      = withdrawal_intensity(seg, scen)
    r      = scen.r0
    spread = r - deposit_rate(seg, scen)
    return (λ, r, spread)
end

function _u0(seg::SegmentParams)
    return [seg.B₀, 1f0, 0f0]
end

function _u0_sa(seg::SegmentParams)
    return @SVector Float32[seg.B₀, 1f0, 0f0]
end

# ---------------------------------------------------------------------------
# CPU batch solve
# ---------------------------------------------------------------------------

"""
    solve_bv_cpu(segments, scen; U=60f0)

Solve the (B,D,V) ODE for each segment sequentially on CPU.
Returns a Vector of EVE values V(U), one per segment.
"""
function solve_bv_cpu(segments::Vector{SegmentParams},
                      scen::CurveScenario;
                      U::Float32=60f0)
    N    = length(segments)
    eves = Vector{Float32}(undef, N)
    for (i, seg) in enumerate(segments)
        p    = _bv_params(seg, scen)
        prob = ODEProblem(bv_ode!, _u0(seg), (0f0, U), p)
        sol  = solve(prob, Tsit5(); save_everystep=false,
                     reltol=1f-6, abstol=1f-8)
        eves[i] = Float32(sol.u[end][3])
    end
    return eves
end

# ---------------------------------------------------------------------------
# GPU batch solve
# ---------------------------------------------------------------------------

"""
    solve_bv_gpu(segments, scen; U=60f0)

GPU-accelerated batch solve via DiffEqGPU EnsembleGPUArray.
Falls back to CPU if CUDA is not available.
Returns a Vector of EVE values V(U), one per segment.

This is where GPU pays off: with thousands of segments × scenarios ×
posterior draws, each tiny 3D ODE runs in its own GPU thread.
"""
function solve_bv_gpu(segments::Vector{SegmentParams},
                      scen::CurveScenario;
                      U::Float32=60f0)
    if !CUDA.functional()
        @warn "CUDA not available — falling back to CPU."
        return solve_bv_cpu(segments, scen; U=U)
    end

    N  = length(segments)
    p0 = _bv_params(segments[1], scen)

    prob = ODEProblem{false}(bv_ode_sa, _u0_sa(segments[1]), (0f0, U), p0)

    function prob_func(prob, i, repeat)
        remake(prob; u0=_u0_sa(segments[i]), p=_bv_params(segments[i], scen))
    end

    ens = EnsembleProblem(prob; prob_func=prob_func)
    sim = solve(ens, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend());
                trajectories=N, save_everystep=false,
                reltol=1f-6, abstol=1f-8)

    return Float32[sim[i].u[end][3] for i in 1:N]
end

end # module
