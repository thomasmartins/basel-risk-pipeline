### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ imports
begin
    include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
    include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
    include(joinpath(@__DIR__, "..", "src", "gpu_ensemble.jl"))
    using .Vasicek
    using .Riccati
    using .GPUEnsemble
    using BenchmarkTools
    using Plots
    using Statistics
    using Printf
    using CUDA
end

# ╔═╡ title
md"""
# 03 — GPU Scaling: When Does It Pay Off?

GPU acceleration is only worthwhile when the workload is **large and parallel**.
A single Riccati ODE solve is trivially fast on CPU — GPU overhead would dominate.

But for Bayesian calibration with $N = 10{,}000$ posterior draws,
each requiring a full ODE solve, the story changes completely.

**Reference:** Utkarsh et al. (2023, arXiv:2304.06835) report
20–100× speedups over JAX/PyTorch for exactly this pattern.
"""

# ╔═╡ cuda_check
begin
    gpu_ok = cuda_available()
    if gpu_ok
        md"✅ CUDA device detected — GPU benchmarks will run."
    else
        md"⚠️ CUDA not available — GPU runs will fall back to CPU. GPU results are indicative only."
    end
end

# ╔═╡ setup
md"""
## Setup: fixed tenors, varying N
"""

# ╔═╡ tenors_def
begin
    tenors  = Float32.(1:30)   # 1–30 year maturities
    Ns      = [10, 100, 500, 1000, 2000, 5000, 10_000]
    nothing
end

# ╔═╡ make_draws
function random_draws(N)
    [ParameterDraw(
        0.3f0 + 0.4f0 * rand(Float32),    # κ ∈ [0.3, 0.7]
        0.01f0 + 0.03f0 * rand(Float32),  # θ ∈ [1%, 4%]
        0.005f0 + 0.015f0 * rand(Float32),# σ ∈ [0.5%, 2%]
        0.02f0 + 0.03f0 * rand(Float32),  # r₀ ∈ [2%, 5%]
    ) for _ in 1:N]
end

# ╔═╡ benchmark_section
md"""
## Benchmark: CPU threads vs GPU ensemble
"""

# ╔═╡ run_benchmarks
begin
    times_cpu = Float64[]
    times_gpu = Float64[]

    for N in Ns
        draws = random_draws(N)

        # CPU
        t_cpu = @elapsed batch_yields_cpu(draws, tenors)
        push!(times_cpu, t_cpu)

        # GPU (or CPU fallback)
        t_gpu = @elapsed batch_yields_gpu(draws, tenors)
        push!(times_gpu, t_gpu)

        @printf "N=%6d  CPU: %.3fs  GPU: %.3fs  speedup: %.1fx\n" N t_cpu t_gpu (t_cpu/t_gpu)
    end
    nothing
end

# ╔═╡ plot_scaling
begin
    pl = plot(Ns, times_cpu,
              label="CPU (EnsembleThreads)",
              xscale=:log10, yscale=:log10,
              marker=:circle, lw=2, color=:steelblue,
              xlabel="N (number of ODE solves)",
              ylabel="Wall time (seconds)",
              title="Riccati ODE ensemble: CPU vs GPU scaling",
              legend=:topleft)

    plot!(pl, Ns, times_gpu,
          label="GPU (EnsembleGPUArray)",
          marker=:square, lw=2, color=:firebrick,
          ls=gpu_ok ? :solid : :dash)

    pl
end

# ╔═╡ plot_speedup
begin
    speedups = times_cpu ./ times_gpu
    bar(Ns, speedups,
        xscale=:log10,
        xlabel="N",
        ylabel="Speedup (CPU / GPU)",
        title="GPU speedup over CPU",
        color=:steelblue,
        legend=false,
        bar_width=0.3)

    hline!([1.0], ls=:dash, color=:black, label="break-even")
end

# ╔═╡ explanation
md"""
## Why the crossover exists

| Regime | Dominant cost | Winner |
|--------|--------------|--------|
| $N < 100$ | GPU kernel launch overhead | CPU |
| $N \approx 1000$ | Break-even | — |
| $N > 1000$ | ODE arithmetic (embarrassingly parallel) | GPU |

The GPU runs **all $N$ trajectories simultaneously** in a single kernel.
CPU threads are limited by core count (typically 8–16).

For Bayesian calibration with $N = 10{,}000$ draws → GPU wins decisively.

→ Next: **`04_bayesian.jl`** — credible bands for NMD rates.
"""
