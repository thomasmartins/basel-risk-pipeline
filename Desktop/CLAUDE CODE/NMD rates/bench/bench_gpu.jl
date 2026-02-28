"""
Benchmark: Julia GPU Riccati ODE ensemble (DiffEqGPU)
======================================================
Run with:  julia --project=.. bench_gpu.jl
Requires CUDA-capable GPU. Falls back to CPU if unavailable.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "vasicek.jl"))
include(joinpath(@__DIR__, "..", "src", "riccati.jl"))
include(joinpath(@__DIR__, "..", "src", "gpu_ensemble.jl"))
using .GPUEnsemble
using BenchmarkTools
using CUDA
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
Ns     = [100, 500, 1_000, 2_000, 5_000, 10_000, 50_000]

println("="^60)
if CUDA.functional()
    println("Julia GPU — DiffEqGPU EnsembleGPUArray")
    println("Device: $(CUDA.name(CUDA.device()))")
else
    println("CUDA not available — running CPU fallback (indicative only)")
end
println("="^60)
@printf "%-8s  %-12s  %-12s  %-8s\n" "N" "CPU(s)" "GPU(s)" "speedup"

for N in Ns
    draws = random_draws(N)

    # warm-up both
    batch_yields_cpu(draws[1:10], tenors)
    batch_yields_gpu(draws[1:10], tenors)

    bm_cpu = @benchmark batch_yields_cpu($draws, $tenors) samples=3 evals=1
    bm_gpu = @benchmark batch_yields_gpu($draws, $tenors) samples=3 evals=1

    t_cpu = median(bm_cpu.times) / 1e9
    t_gpu = median(bm_gpu.times) / 1e9
    @printf "%-8d  %-12.4f  %-12.4f  %-8.1fx\n" N t_cpu t_gpu (t_cpu/t_gpu)
end
