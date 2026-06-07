

-- Per (scenario, date, day-bucket) net cashflow with inflow cap applied.
-- Inflows are scaled by daily_capped_inflows / daily_inflows so each row keeps
-- its proportional share. Outflows pass through negated.

WITH params AS (SELECT * FROM "warehouse"."main"."int_params_wide"),
daily AS (
    SELECT
        scenario_id,
        as_of_date,
        SUM(CASE WHEN direction = 'inflow'  THEN amount ELSE 0 END) AS daily_inflows,
        SUM(CASE WHEN direction = 'outflow' THEN amount ELSE 0 END) AS daily_outflows
    FROM "warehouse"."main"."int_cashflows_enriched"
    GROUP BY scenario_id, as_of_date
),
daily_capped AS (
    SELECT
        d.scenario_id,
        d.as_of_date,
        d.daily_inflows,
        d.daily_outflows,
        LEAST(d.daily_inflows, d.daily_outflows * p.lcr_inflow_cap) AS daily_capped_inflows
    FROM daily d
    CROSS JOIN params p
)
SELECT
    c.scenario_id,
    c.as_of_date,
    c.maturity_day_bucket,
    SUM(
        CASE
            WHEN c.direction = 'outflow' THEN -c.amount
            WHEN c.direction = 'inflow' AND dc.daily_inflows > 0
                THEN c.amount * dc.daily_capped_inflows / dc.daily_inflows
            ELSE 0
        END
    ) AS signed_amount
FROM "warehouse"."main"."int_cashflows_enriched" c
LEFT JOIN daily_capped dc USING (scenario_id, as_of_date)
GROUP BY c.scenario_id, c.as_of_date, c.maturity_day_bucket