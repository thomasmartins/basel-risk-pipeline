{{ config(materialized='view') }}

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
FROM {{ source('risk', 'risk_cbc_ladder') }}
