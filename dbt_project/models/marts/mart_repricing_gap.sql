{{ config(materialized='table') }}

-- Signed repricing gap by EBA-style year bucket (inflow positive, outflow negative).
-- Driver for ∆NII sensitivity in compute.py.

SELECT
    scenario_id,
    maturity_year_bucket AS tenor_bucket,
    SUM(signed_amount) AS gap
FROM {{ ref('int_cashflows_enriched') }}
GROUP BY scenario_id, maturity_year_bucket
