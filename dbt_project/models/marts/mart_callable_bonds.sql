{{ config(materialized='table') }}

-- Per-callable-bond Black-76 decomposition. callable_pv = straight_pv -
-- call_value (the issuer's call option, valued under HW1F closed form).
-- call_value_pct_of_straight gives the "OAS bite" at a glance.

SELECT
    scenario_id,
    cashflow_id,
    notional,
    t_call_years,
    t_mat_years,
    strike_unit,
    integrated_vol,
    straight_pv,
    call_value,
    callable_pv,
    call_value_pct_of_straight
FROM {{ ref('stg_risk_callable_bonds') }}
ORDER BY scenario_id, call_value DESC
