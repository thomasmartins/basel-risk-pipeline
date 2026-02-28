"""
    Bayesian calibration: credible bands for the NMD yield curve
    =============================================================

    Workflow
    --------
    1. Draw N samples from a mock posterior over Vasicek parameters
       (κ, θ, σ, r₀).  In a real application these would come from
       MCMC (e.g. Turing.jl) fitted to observed deposit rates.

    2. For each draw, solve the Riccati ODEs on GPU via GPUEnsemble.
       This is the computational bottleneck DiffEqGPU.jl eliminates.

    3. Aggregate the N yield curves:
         - pointwise median  → posterior median curve
         - pointwise 5th/95th percentile → 90 % credible band

    NMD connection
    --------------
    In the Vasicek model the short rate r(t) proxies the policy rate.
    NMD deposit rate is modelled as a pass-through function:

        d(τ) = α + β · y(τ)

    where y(τ) is the model yield at maturity τ and (α, β) are
    bank-specific pass-through parameters (β ∈ [0,1]).  Uncertainty
    in (κ, θ, σ) propagates into uncertainty in y(τ) and therefore
    into uncertainty in the NMD rate — the credible band captures this.
"""

module BayesCalibration

using Distributions
using Statistics
using ..GPUEnsemble: ParameterDraw, batch_yields_cpu, batch_yields_gpu

export MockPosterior, sample_draws, credible_bands, nmd_rate_bands

# ---------------------------------------------------------------------------
# Mock posterior
# ---------------------------------------------------------------------------

"""
    MockPosterior

Approximate posterior marginals for Vasicek parameters.
In production these would be replaced by MCMC samples.

Typical ECB/Euro-area calibration ranges (annualised):
  κ  ~ 0.3–0.9   (mean reversion ~ 1–3 years)
  θ  ~ 0.01–0.04 (long-run rate ~ 1–4 %)
  σ  ~ 0.005–0.02
  r₀ ~ current short rate, small uncertainty
"""
struct MockPosterior
    κ_dist::Distribution
    θ_dist::Distribution
    σ_dist::Distribution
    r0_dist::Distribution
end

function MockPosterior(;
    κ_mean=0.50f0,  κ_std=0.10f0,
    θ_mean=0.025f0, θ_std=0.005f0,
    σ_mean=0.010f0, σ_std=0.002f0,
    r0_mean=0.035f0, r0_std=0.003f0,
)
    return MockPosterior(
        truncated(Normal(κ_mean, κ_std), 0.05, Inf),
        truncated(Normal(θ_mean, θ_std), 0.001, Inf),
        truncated(Normal(σ_mean, σ_std), 0.001, Inf),
        Normal(r0_mean, r0_std),
    )
end

"""
    sample_draws(post, N) → Vector{ParameterDraw}

Draw N independent samples from the mock posterior.
"""
function sample_draws(post::MockPosterior, N::Int)
    return [ParameterDraw(
                Float32(rand(post.κ_dist)),
                Float32(rand(post.θ_dist)),
                Float32(rand(post.σ_dist)),
                Float32(rand(post.r0_dist)),
            ) for _ in 1:N]
end

# ---------------------------------------------------------------------------
# Credible bands
# ---------------------------------------------------------------------------

"""
    credible_bands(draws, tenors; use_gpu=true, level=0.90)

GPU (or CPU) solve N Riccati systems, return:
  - `median_curve` : pointwise posterior median yield  (length = n_tenors)
  - `lower`        : lower tail  (e.g. 5th percentile for 90% band)
  - `upper`        : upper tail

Returns a NamedTuple with fields `tenors`, `median`, `lower`, `upper`.
"""
function credible_bands(draws::Vector{ParameterDraw},
                        tenors::Vector{Float32};
                        use_gpu::Bool=true,
                        level::Float64=0.90)
    α = (1.0 - level) / 2.0

    yield_mat = use_gpu ?
        batch_yields_gpu(draws, tenors) :
        batch_yields_cpu(draws, tenors)

    # yield_mat is (n_tenors × N)
    med   = vec(mapslices(median, yield_mat; dims=2))
    lower = vec(mapslices(x -> quantile(x, α),       yield_mat; dims=2))
    upper = vec(mapslices(x -> quantile(x, 1.0 - α), yield_mat; dims=2))

    return (tenors=tenors, median=med, lower=lower, upper=upper)
end

# ---------------------------------------------------------------------------
# NMD deposit rate bands
# ---------------------------------------------------------------------------

"""
    nmd_rate_bands(bands; α_pass=0.002, β_pass=0.60)

Apply a simple affine pass-through to convert yield credible bands
into NMD deposit rate credible bands.

    d(τ) = α_pass + β_pass · y(τ)

Default values (α=20 bps, β=0.60) are illustrative.
In practice these are estimated from time-series regressions of
observed deposit rates on market yields.
"""
function nmd_rate_bands(bands; α_pass::Float64=0.002, β_pass::Float64=0.60)
    return (
        tenors = bands.tenors,
        median = α_pass .+ β_pass .* bands.median,
        lower  = α_pass .+ β_pass .* bands.lower,
        upper  = α_pass .+ β_pass .* bands.upper,
    )
end

end # module
