-- LCR inflow cap (EBA Delegated Act 2015/61, Art. 33): capped inflows must not
-- exceed `lcr_inflow_cap * outflows`. Singular test fails if the mart ever
-- violates the cap (modulo float epsilon).

SELECT
    m.scenario_id,
    m.capped_inflows,
    m.outflows,
    p.lcr_inflow_cap,
    m.capped_inflows - m.outflows * p.lcr_inflow_cap AS slack
FROM {{ ref('mart_lcr') }} m
CROSS JOIN {{ ref('int_params_wide') }} p
WHERE m.capped_inflows > m.outflows * p.lcr_inflow_cap + 1e-6
