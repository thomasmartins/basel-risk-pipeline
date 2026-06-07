

-- Daily capital ratios per scenario, smoothed with a 30-day backward
-- rolling window. The synthetic RWA generator scatters ~14 exposures per
-- (date × scenario) on average; raw daily ratios swing widely (CET1 6 %
-- to 60 % within a single scenario) because the *denominator* — daily RWA
-- — has ~25 % relative noise. Real banks don't report daily capital
-- ratios anyway; smoothing reflects the underlying structural ratio and
-- keeps the 4.5 % CET1 / 6 % Tier1 / 8 % Total reference lines visually
-- meaningful on the timeseries chart.

WITH rwa_d AS (
    SELECT as_of_date, scenario_id, SUM(rwa_amount) AS rwa_daily
    FROM "warehouse"."main"."stg_rwa"
    GROUP BY as_of_date, scenario_id
),
joined AS (
    SELECT
        COALESCE(r.as_of_date, b.as_of_date)   AS as_of_date,
        COALESCE(r.scenario_id, b.scenario_id) AS scenario_id,
        r.rwa_daily AS rwa,
        b.cet1, b.tier1, b.total_capital
    FROM rwa_d r
    FULL OUTER JOIN "warehouse"."main"."int_balance_sheet_pivoted" b USING (as_of_date, scenario_id)
)
SELECT
    scenario_id,
    as_of_date,
    AVG(rwa)           OVER w AS rwa,
    AVG(cet1)          OVER w AS cet1,
    AVG(tier1)         OVER w AS tier1,
    AVG(total_capital) OVER w AS total_capital,
    AVG(cet1)          OVER w / NULLIF(AVG(rwa) OVER w, 0) AS cet1_ratio,
    AVG(tier1)         OVER w / NULLIF(AVG(rwa) OVER w, 0) AS tier1_ratio,
    AVG(total_capital) OVER w / NULLIF(AVG(rwa) OVER w, 0) AS total_capital_ratio
FROM joined
WINDOW w AS (
    PARTITION BY scenario_id
    ORDER BY as_of_date
    ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
)