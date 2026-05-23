{{ config(materialized='table') }}

-- Daily LCR using a scenario-wide HQLA stock (matches the static mart's interpretation).
-- Net cashflow column supports the dual-axis bar+line chart.

WITH hqla_per_scn AS (
    SELECT scenario_id, SUM(hqla_post_haircut) AS hqla
    FROM {{ ref('int_cashflows_enriched') }}
    GROUP BY scenario_id
),
daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END) AS inflows,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END) AS outflows
    FROM {{ ref('int_cashflows_enriched') }}
    GROUP BY scenario_id, as_of_date
),
params AS (SELECT * FROM {{ ref('int_params_wide') }})
SELECT
    d.scenario_id,
    d.as_of_date,
    d.inflows,
    d.outflows,
    LEAST(d.inflows, d.outflows * p.lcr_inflow_cap)               AS capped_inflows,
    d.outflows - LEAST(d.inflows, d.outflows * p.lcr_inflow_cap)  AS net_outflows,
    h.hqla,
    h.hqla / NULLIF(d.outflows - LEAST(d.inflows, d.outflows * p.lcr_inflow_cap), 0) AS lcr,
    d.inflows - d.outflows AS net_cashflow
FROM daily d
CROSS JOIN params p
LEFT JOIN hqla_per_scn h ON h.scenario_id = d.scenario_id
