

-- "Daily" LCR with a 30-day backward rolling window, mirroring mart_nsfr_daily.
-- A single-day point ratio is too noisy on this synthetic book; the rolling
-- 30-day view converges to the scenario-wide structural LCR (~1.5-1.9 in the
-- current synthetic data) and keeps the 100 % threshold line visually
-- meaningful.
--
-- Warm-up gate: the rolling window covers up to 30 PRECEDING rows, but the
-- first 29 days of each scenario have an incomplete window. HQLA / inflows /
-- outflows are independently distributed (unlike ASF/RSF in NSFR, which scale
-- together), so partial-window LCR can spike to ~7 or dip below 0.5 purely
-- as a window-fill artefact. We null out the ratio (and the inflow/outflow
-- aggregates) for any row where the window has fewer than 30 rows; the
-- per-day `net_cashflow` column is preserved so the bar overlay still starts
-- on day 1.

WITH daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(hqla_post_haircut)                                       AS daily_hqla,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END)  AS daily_inflows,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END)  AS daily_outflows
    FROM "warehouse"."main"."int_cashflows_enriched"
    GROUP BY scenario_id, as_of_date
),
rolled AS (
    SELECT
        scenario_id,
        as_of_date,
        daily_inflows - daily_outflows                AS net_cashflow,
        COUNT(*)            OVER w                    AS window_rows,
        SUM(daily_hqla)     OVER w                    AS hqla,
        SUM(daily_inflows)  OVER w                    AS inflows,
        SUM(daily_outflows) OVER w                    AS outflows
    FROM daily
    WINDOW w AS (
        PARTITION BY scenario_id
        ORDER BY as_of_date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )
),
params AS (SELECT * FROM "warehouse"."main"."int_params_wide")
SELECT
    r.scenario_id,
    r.as_of_date,
    CASE WHEN r.window_rows < 30 THEN NULL ELSE r.inflows  END AS inflows,
    CASE WHEN r.window_rows < 30 THEN NULL ELSE r.outflows END AS outflows,
    CASE WHEN r.window_rows < 30 THEN NULL ELSE r.hqla     END AS hqla,
    CASE WHEN r.window_rows < 30 THEN NULL
         ELSE LEAST(r.inflows, r.outflows * p.lcr_inflow_cap)
    END AS capped_inflows,
    CASE WHEN r.window_rows < 30 THEN NULL
         ELSE r.outflows - LEAST(r.inflows, r.outflows * p.lcr_inflow_cap)
    END AS net_outflows,
    CASE WHEN r.window_rows < 30 THEN NULL
         ELSE r.hqla / NULLIF(r.outflows - LEAST(r.inflows, r.outflows * p.lcr_inflow_cap), 0)
    END AS lcr,
    r.net_cashflow
FROM rolled r
CROSS JOIN params p