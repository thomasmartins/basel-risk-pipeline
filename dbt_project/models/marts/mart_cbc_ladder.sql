{{ config(materialized='table') }}

-- Daily counterbalancing-capacity trajectory per (scenario, stress).
-- The dashboard plots running_cbc against day_offset; the survival horizon
-- is the first day_offset where running_cbc < 0.

SELECT
    scenario_id,
    stress_name,
    day_offset,
    stressed_inflow,
    stressed_outflow,
    capped_inflow,
    net_cashflow,
    cumulative_net,
    running_cbc
FROM {{ ref('stg_risk_cbc_ladder') }}
ORDER BY scenario_id, stress_name, day_offset
