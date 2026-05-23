{{ config(materialized='table') }}

-- Pass-through: deterministic ΔEVE under the six BCBS 368 prescribed scenarios.

SELECT
    scenario_id,
    shock_scenario,
    delta_eve
FROM {{ ref('stg_risk_eve_bcbs368') }}
