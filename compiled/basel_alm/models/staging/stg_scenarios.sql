

SELECT
    id,
    name AS scenario_name,
    description,
    liquidity_shock AS liquidity_shock_pct,
    ir_shift AS ir_shift_bps,
    credit_shock AS credit_shock_bps
FROM "warehouse"."main"."scenarios"