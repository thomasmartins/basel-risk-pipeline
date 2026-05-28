{{ config(materialized='table') }}

-- "Daily" NSFR with a 30-day backward rolling window. NSFR is a structural
-- (point-in-time, balance-sheet) ratio reported quarterly in practice; this
-- mart keeps a day-by-day series for the trend chart but smooths the
-- single-day noise (per-day NSFR oscillates between ~0.7 and ~3.0 because
-- each day sees a different random slice of the cashflow book). The rolling
-- 30-day view converges quickly to the scenario-wide structural NSFR
-- (~1.22 in the current synthetic data) and makes the 1.00 threshold line
-- visually meaningful.

WITH daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(CASE WHEN asf_factor > 0 THEN asf_contribution ELSE 0 END) AS daily_asf,
        SUM(CASE WHEN rsf_factor > 0 THEN rsf_contribution ELSE 0 END) AS daily_rsf
    FROM {{ ref('int_cashflows_enriched') }}
    GROUP BY scenario_id, as_of_date
)
SELECT
    scenario_id,
    as_of_date,
    SUM(daily_asf) OVER w AS asf,
    SUM(daily_rsf) OVER w AS rsf,
    SUM(daily_asf) OVER w / NULLIF(SUM(daily_rsf) OVER w, 0) AS nsfr
FROM daily
WINDOW w AS (
    PARTITION BY scenario_id
    ORDER BY as_of_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
)
