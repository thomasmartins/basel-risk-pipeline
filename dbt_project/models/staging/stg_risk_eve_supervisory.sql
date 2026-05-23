{{ config(materialized='view') }}

SELECT
    scenario_id,
    worst_scenario,
    worst_delta_eve,
    tier1_capital,
    ratio,
    breach,
    distributional_99
FROM {{ source('risk', 'risk_eve_supervisory') }}
