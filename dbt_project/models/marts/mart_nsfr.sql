{{ config(materialized='table') }}

-- NSFR per scenario. ASF / RSF aggregate rows with non-zero asf_factor / rsf_factor
-- respectively (direction is not what determines the balance-sheet side).

SELECT
    scenario_id,
    SUM(CASE WHEN asf_factor > 0 THEN asf_contribution ELSE 0 END) AS asf,
    SUM(CASE WHEN rsf_factor > 0 THEN rsf_contribution ELSE 0 END) AS rsf,
    SUM(CASE WHEN asf_factor > 0 THEN asf_contribution ELSE 0 END) /
        NULLIF(SUM(CASE WHEN rsf_factor > 0 THEN rsf_contribution ELSE 0 END), 0) AS nsfr
FROM {{ ref('int_cashflows_enriched') }}
GROUP BY scenario_id
