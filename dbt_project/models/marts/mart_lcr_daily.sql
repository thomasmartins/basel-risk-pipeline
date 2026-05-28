{{ config(materialized='table') }}

-- Daily LCR series for the timeseries chart.
--
-- The synthetic cashflow book is stylised as a single LCR stress horizon
-- aggregated over the whole sample (mart_lcr does the same); reusing those
-- per-scenario aggregates here keeps daily LCR consistent with the static
-- KPI — both land in the realistic 1.5-1.9 band (150-190 %) so the 100 %
-- threshold line on the chart is visually meaningful.
--
-- The daily variation in the chart comes from the `net_cashflow` bar
-- overlay (daily inflows − outflows), not from the LCR ratio itself,
-- which is structural at the scenario level.

WITH scenario_totals AS (
    SELECT
        scenario_id,
        SUM(hqla_post_haircut)                                       AS hqla,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END)  AS outflows,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END)  AS inflows
    FROM {{ ref('int_cashflows_enriched') }}
    GROUP BY scenario_id
),
daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END) AS daily_inflows,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END) AS daily_outflows
    FROM {{ ref('int_cashflows_enriched') }}
    GROUP BY scenario_id, as_of_date
),
params AS (SELECT * FROM {{ ref('int_params_wide') }})
SELECT
    d.scenario_id,
    d.as_of_date,
    s.inflows,
    s.outflows,
    LEAST(s.inflows, s.outflows * p.lcr_inflow_cap)               AS capped_inflows,
    s.outflows - LEAST(s.inflows, s.outflows * p.lcr_inflow_cap)  AS net_outflows,
    s.hqla,
    s.hqla / NULLIF(s.outflows - LEAST(s.inflows, s.outflows * p.lcr_inflow_cap), 0) AS lcr,
    d.daily_inflows - d.daily_outflows AS net_cashflow
FROM daily d
JOIN scenario_totals s ON s.scenario_id = d.scenario_id
CROSS JOIN params p
