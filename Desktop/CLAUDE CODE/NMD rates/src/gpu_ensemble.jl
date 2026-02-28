"""
    GPU-accelerated ensemble of Riccati ODE solves
    ================================================

    When we have N parameter draws (e.g. from a posterior), we need to
    solve the Riccati ODE N times — once per draw. This is an embarrassingly
    parallel problem and exactly the use-case DiffEqGPU.jl is designed for.

    From the paper (Utkarsh et al. 2023, arXiv:2304.06835):
      "EnsembleGPUArray ... translates the ODE to GPU kernel code automatically,
       achieving 20–100× speedup over vectorised JAX/PyTorch approaches."

    Key insight:
      - CPU (EnsembleThreads): parallelise over CPU cores. Fine for N < ~1000.
      - GPU (EnsembleGPUArray): run thousands of tiny ODE solves in one kernel.
        GPU wins only when N is large enough to saturate the device
        (typically N ≥ 1000 for modern GPUs).

    StaticArrays requirement:
      DiffEqGPU kernels cannot use heap-allocated arrays. We rewrite the
      ODE with SVector (stack-allocated, size known at compile time).
"""

module GPUEnsemble

using OrdinaryDiffEq
using DiffEqGPU
using StaticArrays
using CUDA

export ParameterDraw, batch_yields_cpu, batch_yields_gpu, cuda_available

# ---------------------------------------------------------------------------
# Parameter draw type
# ---------------------------------------------------------------------------

"""
A single draw from the posterior (or any parameter set).
All fields are Float32 for GPU compatibility.
"""
struct ParameterDraw
    κ::Float32
    θ::Float32
    σ::Float32
    x0::Float32
end

# ---------------------------------------------------------------------------
# ODE with StaticArrays  (required for GPU kernels)
# ---------------------------------------------------------------------------

"""
    riccati_static(u, params, τ)

Out-of-place Riccati ODE using SVector.
params = (κ, θ, σ²)  — Vasicek special case (α₁ = 0, δ₀ = 0, δ₁ = 1).
"""
function riccati_static(u::SVector{2,T}, params, τ) where T
    B, A = u
    κ, θ, σ² = params
    dB = one(T) - κ * B
    dA = -κ * θ * B + (σ² / 2) * B^2
    return SVector{2,T}(dB, dA)
end

# ---------------------------------------------------------------------------
# Ensemble problem factory
# ---------------------------------------------------------------------------

function make_ensemble_problem(draws::Vector{ParameterDraw}, τ_max::Float32)
    # Base problem (params will be overridden per draw)
    u0     = @SVector Float32[0f0, 0f0]
    p0     = (draws[1].κ, draws[1].θ, draws[1].σ^2)
    tspan  = (0f0, τ_max)
    prob   = ODEProblem{false}(riccati_static, u0, tspan, p0)

    function prob_func(prob, i, repeat)
        d = draws[i]
        remake(prob; p=(d.κ, d.θ, d.σ^2f0))
    end

    return EnsembleProblem(prob; prob_func=prob_func)
end

# ---------------------------------------------------------------------------
# Yield extraction helper
# ---------------------------------------------------------------------------

function _extract_yields(sol_i, tenors::Vector{Float32}, x0::Float32)
    [begin
        B, A = sol_i(τ)
        (-A + B * x0) / τ
     end for τ in tenors]
end

# ---------------------------------------------------------------------------
# CPU batch solve
# ---------------------------------------------------------------------------

"""
    batch_yields_cpu(draws, tenors; trajectories=nothing)

Solve Riccati ODEs for all `draws` in parallel on CPU threads.
Returns a Matrix of size (n_tenors × n_draws).
"""
function batch_yields_cpu(draws::Vector{ParameterDraw},
                          tenors::Vector{Float32};
                          saveat::Vector{Float32}=tenors)
    τ_max  = maximum(tenors)
    ens    = make_ensemble_problem(draws, τ_max)
    N      = length(draws)

    sim = solve(ens, Tsit5(), EnsembleThreads();
                trajectories=N, saveat=saveat,
                reltol=1f-6, abstol=1f-8)

    out = Matrix{Float32}(undef, length(tenors), N)
    for i in 1:N
        out[:, i] = _extract_yields(sim[i], tenors, draws[i].x0)
    end
    return out
end

# ---------------------------------------------------------------------------
# GPU batch solve
# ---------------------------------------------------------------------------

"""
    batch_yields_gpu(draws, tenors)

Solve Riccati ODEs for all `draws` on GPU (if available) via EnsembleGPUArray.
Falls back to EnsembleThreads with a warning if CUDA is not functional.

Returns a Matrix of size (n_tenors × n_draws).
"""
function batch_yields_gpu(draws::Vector{ParameterDraw},
                          tenors::Vector{Float32};
                          saveat::Vector{Float32}=tenors)
    if !CUDA.functional()
        @warn "CUDA not available — falling back to CPU ensemble."
        return batch_yields_cpu(draws, tenors; saveat=saveat)
    end

    τ_max = maximum(tenors)
    ens   = make_ensemble_problem(draws, τ_max)
    N     = length(draws)

    sim = solve(ens, Tsit5(), EnsembleGPUArray(CUDA.CUDABackend());
                trajectories=N, saveat=saveat,
                reltol=1f-6, abstol=1f-8)

    out = Matrix{Float32}(undef, length(tenors), N)
    for i in 1:N
        out[:, i] = _extract_yields(sim[i], tenors, draws[i].x0)
    end
    return out
end

"""
    cuda_available()

Check and print CUDA status.
"""
function cuda_available()
    if CUDA.functional()
        println("CUDA functional: ", CUDA.name(CUDA.device()))
        return true
    else
        println("CUDA not available — GPU solves will fall back to CPU.")
        return false
    end
end

end # module
