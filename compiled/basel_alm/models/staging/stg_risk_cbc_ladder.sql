

SELECT
    scenario_id,
    stress_name,
    day_offset,
    stressed_inflow,
    stressed_outflow,
    capped_inflow,
    net_cashflow,
    cumulative_net,
    running_cbc
FROM "warehouse"."main"."risk_cbc_ladder"