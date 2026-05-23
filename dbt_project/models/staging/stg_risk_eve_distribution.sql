{{ config(materialized='view') }}

SELECT
    scenario_id,
    path_id,
    delta_eve
FROM {{ source('risk', 'risk_eve_distribution') }}
