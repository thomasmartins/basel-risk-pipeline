"""
    NMD Valuation: shocked curves, EVE portfolio, duration, IRRBB shock table
    ==========================================================================

    Builds on NMDSegments to compute portfolio-level outputs:

    - EVE of each segment under any CurveScenario
    - Shocked yield curves (parallel ±, steepener, flattener, short-end ±)
      consistent with Basel IRRBB standard scenarios (BCBS 368)
    - Effective duration: -dEVE/dy / EVE  (numerical, 1bp bump)
    - Full IRRBB shock table across all segments and all scenarios
"""

module NMDValuation

using ..NMDSegments
using ..Vasicek: VasicekParams, yield_curve as vasicek_yields

export CurveScenario                       # re-export for convenience
export base_scenario, shocked_curve
export eve_portfolio
export effective_duration
export shock_table, IRRBB_SHOCKS

# ---------------------------------------------------------------------------
# Canonical IRRBB shock labels
# ---------------------------------------------------------------------------

const IRRBB_SHOCKS = [
    :parallel_up,
    :parallel_down,
    :steepener,
    :flattener,
    :short_end_up,
    :short_end_down,
]

# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

"""
    base_scenario(vp, tenors)

Build a CurveScenario from a VasicekParams calibration.
"""
function base_scenario(vp::VasicekParams, tenors::Vector{Float32})
    yields = Float32.(vasicek_yields(vp, Float64.(tenors)))
    return CurveScenario(Float32(vp.r0), yields, tenors, "base")
end

"""
    shocked_curve(base, shock_type, magnitude)

Apply a deterministic rate shock to `base` and return a new CurveScenario.

Shock types (consistent with BCBS 368 spirit):
  :parallel_up / :parallel_down  — uniform shift of ±magnitude
  :steepener    — short rates down, long rates up  (slope +2·magnitude)
  :flattener    — short rates up, long rates down  (slope -2·magnitude)
  :short_end_up / :short_end_down — exponentially decaying shock at short end

`magnitude` is in annual decimal (0.02 = 200 bp).

Note: for a single-factor model (Vasicek) all curve movements reduce to
parallel shifts in r₀. The non-parallel shapes here are applied additively
to the yield curve and do NOT need to be model-consistent — they represent
regulatory scenarios, not model-implied dynamics.
"""
function shocked_curve(base::CurveScenario,
                       shock_type::Symbol,
                       magnitude::Float32)
    (; r0, yields, tenors, label) = base
    n     = length(tenors)
    shift = zeros(Float32, n)
    τ_max = tenors[end]

    if shock_type == :parallel_up
        shift .= magnitude
        r0_new = r0 + magnitude

    elseif shock_type == :parallel_down
        shift .= -magnitude
        r0_new = r0 - magnitude

    elseif shock_type == :steepener
        # Short end down by magnitude, long end up by magnitude
        for (i, τ) in enumerate(tenors)
            shift[i] = magnitude * (2f0 * τ / τ_max - 1f0)
        end
        r0_new = r0 - magnitude   # short rate goes down

    elseif shock_type == :flattener
        # Short end up by magnitude, long end down by magnitude
        for (i, τ) in enumerate(tenors)
            shift[i] = magnitude * (1f0 - 2f0 * τ / τ_max)
        end
        r0_new = r0 + magnitude

    elseif shock_type == :short_end_up
        # Exponential decay: full shock at τ→0, zero by ~5Y
        for (i, τ) in enumerate(tenors)
            shift[i] = magnitude * exp(-τ / 2f0)
        end
        r0_new = r0 + magnitude

    elseif shock_type == :short_end_down
        for (i, τ) in enumerate(tenors)
            shift[i] = -magnitude * exp(-τ / 2f0)
        end
        r0_new = r0 - magnitude

    else
        error("Unknown shock type: $shock_type. Choose from $IRRBB_SHOCKS")
    end

    new_yields = yields .+ shift
    return CurveScenario(r0_new, new_yields, tenors, string(shock_type))
end

# ---------------------------------------------------------------------------
# Portfolio EVE
# ---------------------------------------------------------------------------

"""
    eve_portfolio(segments, scen; use_gpu=false, U=60f0)

Compute EVE for each segment under scenario `scen`.
Returns a Vector{Float32} of length n_segments.
"""
function eve_portfolio(segments::Vector{SegmentParams},
                       scen::CurveScenario;
                       use_gpu::Bool=false,
                       U::Float32=60f0)
    return use_gpu ?
        solve_bv_gpu(segments, scen; U=U) :
        solve_bv_cpu(segments, scen; U=U)
end

# ---------------------------------------------------------------------------
# Effective duration
# ---------------------------------------------------------------------------

"""
    effective_duration(segments, scen; Δ=1f-4, U=60f0)

Numerical effective duration per segment (parallel bump-and-reprice):

    D_eff = -(V_up - V_dn) / (2Δ · V_base)

Δ = 1 bp (0.0001) by default.
Positive duration means EVE falls when rates rise.
"""
function effective_duration(segments::Vector{SegmentParams},
                             scen::CurveScenario;
                             Δ::Float32=1f-4,
                             U::Float32=60f0)
    scen_up = shocked_curve(scen, :parallel_up,   Δ)
    scen_dn = shocked_curve(scen, :parallel_down, Δ)

    eve_base = solve_bv_cpu(segments, scen;    U=U)
    eve_up   = solve_bv_cpu(segments, scen_up; U=U)
    eve_dn   = solve_bv_cpu(segments, scen_dn; U=U)

    # Avoid division by zero for near-zero EVE
    durations = map(1:length(segments)) do i
        abs(eve_base[i]) < 1f-10 && return 0f0
        -(eve_up[i] - eve_dn[i]) / (2f0 * Δ * eve_base[i])
    end
    return Float32.(durations)
end

# ---------------------------------------------------------------------------
# Full IRRBB shock table
# ---------------------------------------------------------------------------

"""
    shock_table(segments, vp, tenors; magnitude=0.02f0, use_gpu=false, U=60f0)

Compute EVE for all segments under the base scenario and all 6 IRRBB shocks.

Returns a NamedTuple with:
  - `scenarios`: Vector of scenario labels (base + 6 shocks)
  - `eve`:       Matrix{Float32} of size (n_segments × n_scenarios)
  - `eve_change`: Matrix{Float32} — change from base (regulatory metric)
  - `duration`:  Vector{Float32} — effective duration per segment
  - `segment_names`: Vector{String}

`magnitude` = 0.02 corresponds to the standard ±200 bp BCBS shock.
"""
function shock_table(segments::Vector{SegmentParams},
                     vp::VasicekParams,
                     tenors::Vector{Float32};
                     magnitude::Float32=0.02f0,
                     use_gpu::Bool=false,
                     U::Float32=60f0)
    base   = base_scenario(vp, tenors)
    shocks = [base; [shocked_curve(base, s, magnitude) for s in IRRBB_SHOCKS]]
    labels = [sc.label for sc in shocks]
    n_seg  = length(segments)
    n_scen = length(shocks)

    eve_mat = Matrix{Float32}(undef, n_seg, n_scen)
    for (j, scen) in enumerate(shocks)
        eve_mat[:, j] = eve_portfolio(segments, scen; use_gpu=use_gpu, U=U)
    end

    eve_change = eve_mat .- eve_mat[:, 1]   # delta vs base (col 1)
    dur        = effective_duration(segments, base; U=U)

    return (
        scenarios     = labels,
        eve           = eve_mat,
        eve_change    = eve_change,
        duration      = dur,
        segment_names = [s.name for s in segments],
    )
end

end # module
