{{ config(materialized='table') }}

-- PV01 by tenor bucket, derived from the actual cashflow book — replaces
-- the white-noise irrbb.pv01 column. For each row,
--   PV01 = sign · amount · tau · DF(tau) · 0.0001
-- with sign = +1 for assets (loan / bond), -1 for liabilities (deposit),
-- tau = maturity_days / 365, and DF a closed-form approximation at a flat
-- 3 % continuously compounded rate (the average level of the synthetic base
-- curve). The risk engine uses the full HW1F-implied curve for the BCBS 368
-- ΔEVE path; this mart is the curve-light dashboard-side view.

WITH priced AS (
    SELECT
        scenario_id,
        maturity_year_bucket AS tenor_bucket,
        CASE WHEN product IN ('loan', 'bond') THEN 1.0 ELSE -1.0 END
            * amount
            * (maturity_days / 365.0)
            * EXP(-0.03 * (maturity_days / 365.0))
            * 0.0001 AS pv01_contribution
    FROM {{ ref('int_cashflows_enriched') }}
)
SELECT
    scenario_id,
    tenor_bucket,
    SUM(pv01_contribution) AS pv01
FROM priced
GROUP BY scenario_id, tenor_bucket
