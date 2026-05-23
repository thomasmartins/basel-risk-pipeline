{{ config(materialized='view') }}

SELECT
    scenario_id,
    shock_scenario,
    delta_eve
FROM {{ source('risk', 'risk_eve_bcbs368') }}
