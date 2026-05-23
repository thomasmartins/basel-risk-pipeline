{{ config(materialized='table') }}

-- ASF and RSF contribution broken down by product, for the funding-structure bar chart.

SELECT
    scenario_id,
    product,
    'ASF' AS side,
    SUM(asf_contribution) AS contribution,
    AVG(asf_factor)       AS avg_factor
FROM {{ ref('int_cashflows_enriched') }}
WHERE asf_factor > 0
GROUP BY scenario_id, product

UNION ALL

SELECT
    scenario_id,
    product,
    'RSF' AS side,
    SUM(rsf_contribution) AS contribution,
    AVG(rsf_factor)       AS avg_factor
FROM {{ ref('int_cashflows_enriched') }}
WHERE rsf_factor > 0
GROUP BY scenario_id, product
