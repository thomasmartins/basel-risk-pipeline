

-- LCR per scenario. Net outflows = outflows - LEAST(inflows, outflows * inflow_cap).
-- Inflow cap default 75% per EBA Delegated Act.

WITH base AS (
    SELECT
        scenario_id,
        SUM(hqla_post_haircut)                                       AS hqla,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END)  AS outflows,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END)  AS inflows
    FROM "warehouse"."main"."int_cashflows_enriched"
    GROUP BY scenario_id
),
params AS (SELECT * FROM "warehouse"."main"."int_params_wide")
SELECT
    b.scenario_id,
    b.hqla,
    b.outflows,
    b.inflows,
    LEAST(b.inflows, b.outflows * p.lcr_inflow_cap)               AS capped_inflows,
    b.outflows - LEAST(b.inflows, b.outflows * p.lcr_inflow_cap)  AS net_outflows,
    b.hqla / NULLIF(b.outflows - LEAST(b.inflows, b.outflows * p.lcr_inflow_cap), 0) AS lcr
FROM base b
CROSS JOIN params p