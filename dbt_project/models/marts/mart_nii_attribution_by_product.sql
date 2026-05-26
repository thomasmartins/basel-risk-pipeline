{{ config(materialized='table') }}

-- Product-level NII attribution: sum the per-row components by product per scenario.

SELECT
    scenario_id,
    product,
    SUM(customer_margin)   AS customer_margin,
    SUM(funding_margin)    AS funding_margin,
    SUM(behavioral_value)  AS behavioral_value,
    SUM(nii_total)         AS nii_total,
    COUNT(*)               AS n_rows
FROM {{ ref('stg_risk_nii_attribution_rows') }}
GROUP BY scenario_id, product
ORDER BY scenario_id, product
