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
    include(joinpath(@__DIR__, "..", "src", "nmd_valuation.jl"))
    using .Vasicek
    using .Riccati
    using .NMDSegments
    using .NMDValuation
    using Plots
    using Printf
    using Statistics
end

# ╔═╡ title
md"""
# 06 — NMD Portfolio EVE: Multi-Segment IRRBB Shock Table

Full pipeline:
1. Calibrate yield curve (Vasicek Riccati ODE)
2. Define multi-segment NMD portfolio
3. Compute EVE under 6 Basel IRRBB scenarios
4. Effective duration per segment
"""

# ╔═╡ curve_setup
md"""
## Step 1 — Yield curve (Vasicek)
"""

# ╔═╡ vasicek_setup
begin
    vp     = VasicekParams(0.50, 0.025, 0.010, 0.035)
    tenors = Float32.(1:30)
    base   = base_scenario(vp, tenors)
    @printf "Short rate r₀:      %.2f%%\n" base.r0*100
    @printf "2Y yield:           %.2f%%\n" yield_at(base, 2f0)*100
    @printf "10Y yield:          %.2f%%\n" yield_at(base, 10f0)*100
    @printf "Long-run yield:     %.2f%%\n" yield_at(base, 30f0)*100
    nothing
end

# ╔═╡ plot_base_curve
begin
    plot(tenors, base.yields .* 100,
         xlabel="Maturity (years)", ylabel="Yield (%)",
         title="Base yield curve (Vasicek)",
         lw=2, color=:steelblue, legend=false)
end

# ╔═╡ segments_setup
md"""
## Step 2 — NMD portfolio: four segments

| Segment | β | λ₀ | γ | Anchor |
|---------|---|-----|---|--------|
| Retail current | 0.20 | 5%/yr | 1.0 | 3M (~0.25Y) |
| Retail savings | 0.40 | 10%/yr | 2.0 | 2Y |
| SME operating  | 0.30 | 15%/yr | 1.5 | 1Y |
| Corporate      | 0.60 | 25%/yr | 3.0 | 5Y |

Pass-through reflects stickiness: current accounts (β=0.20) barely move;
corporate deposits (β=0.60) are more market-sensitive.
"""

# ╔═╡ define_segments
begin
    segments = [
        SegmentParams("Retail current",
            -0.002f0, 0.20f0, 0.25f0, 1.0f0, 0.05f0, 1.0f0, 1.0f0),
        SegmentParams("Retail savings",
            -0.003f0, 0.40f0, 2.0f0,  5.0f0, 0.10f0, 2.0f0, 1.0f0),
        SegmentParams("SME operating",
            -0.001f0, 0.30f0, 1.0f0,  2.0f0, 0.15f0, 1.5f0, 1.0f0),
        SegmentParams("Corporate",
             0.000f0, 0.60f0, 5.0f0, 10.0f0, 0.25f0, 3.0f0, 1.0f0),
    ]

    # Base deposit rates and run-off intensities
    println("Segment diagnostics (base scenario):")
    println("─"^62)
    @printf "%-20s  %6s  %6s  %6s  %6s\n" "Segment" "r_dep%" "spread%" "λ/yr" "EVE"
    println("─"^62)
    base_eves = solve_bv_cpu(segments, base)
    for (seg, eve) in zip(segments, base_eves)
        rd  = deposit_rate(seg, base)
        spr = base.r0 - rd
        λ   = withdrawal_intensity(seg, base)
        @printf "%-20s  %6.2f  %6.2f  %6.2f  %6.4f\n" seg.name rd*100 spr*100 λ eve
    end
    println("─"^62)
    @printf "%-20s  %6s  %6s  %6s  %6.4f\n" "TOTAL" "" "" "" sum(base_eves)
    nothing
end

# ╔═╡ shock_scenarios
md"""
## Step 3 — IRRBB shock table (Basel 6 scenarios, ±200bp)
"""

# ╔═╡ compute_shocks
begin
    tbl = shock_table(segments, vp, tenors; magnitude=0.02f0)
    nothing
end

# ╔═╡ print_shock_table
begin
    println("EVE by segment and scenario (magnitude = 200 bp)")
    println("─"^90)
    header = @sprintf "%-20s" "Segment"
    for sc in tbl.scenarios
        header *= @sprintf "  %12s" sc
    end
    println(header)
    println("─"^90)
    for i in 1:length(segments)
        row = @sprintf "%-20s" tbl.segment_names[i]
        for j in 1:length(tbl.scenarios)
            row *= @sprintf "  %12.4f" tbl.eve[i,j]
        end
        println(row)
    end
    println("─"^90)
    totrow = @sprintf "%-20s" "TOTAL"
    for j in 1:length(tbl.scenarios)
        totrow *= @sprintf "  %12.4f" sum(tbl.eve[:,j])
    end
    println(totrow)
    nothing
end

# ╔═╡ plot_eve_change
begin
    n_seg  = length(segments)
    n_scen = length(tbl.scenarios) - 1   # exclude base

    shock_labels = tbl.scenarios[2:end]
    eve_Δ        = tbl.eve_change[:, 2:end]   # rows=segments, cols=shocks

    colors = [:steelblue, :firebrick, :darkorange, :seagreen,
              :mediumpurple, :saddlebrown]

    pl = plot(title="ΔEVE by scenario (vs base)",
              xlabel="Scenario", ylabel="ΔEVE (per unit balance)",
              xticks=(1:n_scen, shock_labels),
              xrotation=25, legend=:outertopright,
              size=(850, 400))

    for (i, seg) in enumerate(segments)
        plot!(pl, 1:n_scen, eve_Δ[i, :],
              label=seg.name, marker=:circle, lw=2, color=colors[i])
    end

    hline!(pl, [0], ls=:dash, color=:black, lw=1, label="")
    pl
end

# ╔═╡ plot_shocked_curves
begin
    shock_types  = NMDValuation.IRRBB_SHOCKS
    shock_colors = [:firebrick, :steelblue, :darkorange, :seagreen,
                    :mediumpurple, :saddlebrown]

    pl2 = plot(tenors, base.yields .* 100,
               label="base", lw=2.5, color=:black,
               xlabel="Maturity (years)", ylabel="Yield (%)",
               title="Yield curves under IRRBB shocks (±200 bp)")

    for (shock, col) in zip(shock_types, shock_colors)
        scen = shocked_curve(base, shock, 0.02f0)
        plot!(pl2, tenors, scen.yields .* 100,
              label=string(shock), lw=1.5, ls=:dash, color=col)
    end
    pl2
end

# ╔═╡ duration_section
md"""
## Step 4 — Effective duration

Duration = $-\frac{dV}{dy} \cdot \frac{1}{V}$ (parallel bump, 1 bp)

Positive duration means EVE falls when rates rise.
Segments with higher β or higher γ tend to have **shorter duration**:
pass-through deposits lose less EVE in a rate-up scenario because their
funding cost rises with the market, preserving the spread.
"""

# ╔═╡ compute_duration
begin
    durations = tbl.duration
    println("Effective duration by segment:")
    println("─"^40)
    for (seg, dur) in zip(segments, durations)
        @printf "  %-20s  %6.2f yrs\n" seg.name dur
    end
    println("─"^40)
    @printf "  %-20s  %6.2f yrs\n" "Portfolio avg" mean(durations)
    nothing
end

# ╔═╡ plot_duration
begin
    seg_names = [s.name for s in segments]
    bar(seg_names, durations,
        xlabel="Segment", ylabel="Effective duration (years)",
        title="NMD effective duration\n(parallel +1bp bump-and-reprice)",
        color=[:steelblue, :firebrick, :darkorange, :seagreen],
        legend=false, xrotation=15)
    hline!([0], ls=:dash, color=:black, lw=1)
end

# ╔═╡ summary_section
md"""
## Summary

| Output | Description |
|--------|-------------|
| **EVE** | PV of spread earned on remaining balance under static curve |
| **ΔEVE** | EVE change under IRRBB shock — regulatory sensitivity metric |
| **Duration** | Effective rate sensitivity of NMD franchise value |

**Key observations for this portfolio:**
- High-pass-through segments (corporate, β=0.60) have **lower EVE** in the base:
  the spread is eroded by the market-linked deposit rate.
- Under **parallel up**: sticky deposits (retail current, β=0.20) take the
  largest EVE hit — their deposit rate barely rises, but the discount rate jumps.
- **Steepener** shock: segments anchored at the short end gain (short rates down
  → lower deposit cost), while long-anchored segments lose.
- **Duration** is shorter for high-γ segments because customers leave faster when
  rates rise, shortening the effective maturity of the balance sheet.

→ This is the valuation backbone for a full NII/EVE IRRBB model.
"""
