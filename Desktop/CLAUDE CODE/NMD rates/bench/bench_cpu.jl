"""
Benchmark: Julia CPU Riccati ODE ensemble
=========================================
Run with:  julia --project=.. bench_cpu.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_ensemble.jl"))
using .GPUEnsemble
using BenchmarkTools
using Printf

function random_draws(N)
    [ParameterDraw(
        0.3f0 + 0.4f0 * rand(Float32),
        0.01f0 + 0.03f0 * rand(Float32),
        0.005f0 + 0.015f0 * rand(Float32),
        0.02f0 + 0.03f0 * rand(Float32),
    ) for _ in 1:N]
end

tenors = Float32.(1:30)
Ns     = [10, 100, 500, 1_000, 2_000, 5_000, 10_000]

println("="^55)
println("Julia CPU — Riccati ODE ensemble (EnsembleThreads)")
println("Threads: $(Threads.nthreads())")
println("="^55)
@printf "%-8s  %-10s  %-10s\n" "N" "median(s)" "allocs"

for N in Ns
    draws = random_draws(N)
    # warm-up
    batch_yields_cpu(draws[1:min(N,10)], tenors)

    bm = @benchmark batch_yields_cpu($draws, $tenors) samples=5 evals=1
    @printf "%-8d  %-10.4f  %-10d\n" N median(bm.times)/1e9 bm.allocs
end
