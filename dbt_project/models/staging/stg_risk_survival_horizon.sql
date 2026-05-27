{{ config(materialized='view') }}

SELECT
    scenario_id,
    stress_name,
    initial_cbc,
    survival_horizon_days,
    is_breached,
    peak_deficit
FROM {{ source('risk', 'risk_survival_horizon') }}
