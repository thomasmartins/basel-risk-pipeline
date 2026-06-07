

-- "Daily" NSFR with a 30-day backward rolling window. NSFR is a structural
-- (point-in-time, balance-sheet) ratio reported quarterly in practice; this
-- mart keeps a day-by-day series for the trend chart but smooths the
-- single-day noise (per-day NSFR oscillates between ~0.7 and ~3.0 because
-- each day sees a different random slice of the cashflow book). The rolling
-- 30-day view converges quickly to the scenario-wide structural NSFR
-- (~1.22 in the current synthetic data) and makes the 1.00 threshold line
-- visually meaningful.
--
-- Warm-up gate: matches mart_lcr_daily — null the ratio for any row where
-- the rolling window has fewer than 30 rows of data. NSFR's warm-up is much
-- milder than LCR's (ASF/RSF scale together by construction) but the gate is
-- kept symmetric so the two charts have the same effective start date.

WITH daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(CASE WHEN asf_factor > 0 THEN asf_contribution ELSE 0 END) AS daily_asf,
        SUM(CASE WHEN rsf_factor > 0 THEN rsf_contribution ELSE 0 END) AS daily_rsf
    FROM "warehouse"."main"."int_cashflows_enriched"
    GROUP BY scenario_id, as_of_date
),
rolled AS (
    SELECT
        scenario_id,
        as_of_date,
        COUNT(*)         OVER w AS window_rows,
        SUM(daily_asf)   OVER w AS asf,
        SUM(daily_rsf)   OVER w AS rsf
    FROM daily
    WINDOW w AS (
        PARTITION BY scenario_id
        ORDER BY as_of_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )
)
SELECT
    scenario_id,
    as_of_date,
    CASE WHEN window_rows < 30 THEN NULL ELSE asf END AS asf,
    CASE WHEN window_rows < 30 THEN NULL ELSE rsf END AS rsf,
    CASE WHEN window_rows < 30 THEN NULL ELSE asf / NULLIF(rsf, 0) END AS nsfr
FROM rolled