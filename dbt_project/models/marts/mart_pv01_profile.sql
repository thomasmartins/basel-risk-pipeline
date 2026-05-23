{{ config(materialized='table') }}

SELECT
    scenario_id,
    tenor_bucket,
    SUM(pv01) AS pv01
FROM {{ ref('stg_irrbb') }}
GROUP BY scenario_id, tenor_bucket
