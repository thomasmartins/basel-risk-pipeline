

SELECT
    scenario_id,
    cashflow_id,
    notional,
    t_call_years,
    t_mat_years,
    strike_unit,
    integrated_vol,
    straight_pv,
    call_value,
    callable_pv,
    CASE
        WHEN straight_pv > 0 THEN call_value / straight_pv
        ELSE 0
    END AS call_value_pct_of_straight
FROM "warehouse"."main"."risk_callable_bonds"