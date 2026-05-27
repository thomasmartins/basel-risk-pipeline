{{ config(materialized='table') }}

-- Per-scenario optionality summary: combines mortgage CPR impact and callable
-- bond call-value drag into one row. Useful for headline numbers on the
-- dashboard ("how much PV is the option deck costing us this scenario").

WITH mortgage_totals AS (
    SELECT
        scenario_id,
        COUNT(*) AS n_mortgages,
        SUM(notional) AS mortgage_notional,
        SUM(pv_cpr) AS mortgage_pv_cpr,
        SUM(pv_scheduled) AS mortgage_pv_scheduled,
        SUM(pv_cpr_impact) AS mortgage_pv_cpr_impact,
        AVG(avg_cpr) AS mortgage_avg_cpr,
        AVG(weighted_avg_life_years) AS mortgage_avg_wal_years
    FROM {{ ref('stg_risk_mortgage_cashflows') }}
    GROUP BY scenario_id
),
callable_totals AS (
    SELECT
        scenario_id,
        COUNT(*) AS n_callable_bonds,
        SUM(notional) AS callable_notional,
        SUM(straight_pv) AS callable_straight_pv,
        SUM(call_value) AS callable_call_value,
        SUM(callable_pv) AS callable_callable_pv,
        AVG(integrated_vol) AS callable_avg_integrated_vol
    FROM {{ ref('stg_risk_callable_bonds') }}
    GROUP BY scenario_id
)
SELECT
    s.id AS scenario_id,
    COALESCE(m.n_mortgages, 0) AS n_mortgages,
    COALESCE(m.mortgage_notional, 0) AS mortgage_notional,
    COALESCE(m.mortgage_pv_cpr, 0) AS mortgage_pv_cpr,
    COALESCE(m.mortgage_pv_scheduled, 0) AS mortgage_pv_scheduled,
    COALESCE(m.mortgage_pv_cpr_impact, 0) AS mortgage_pv_cpr_impact,
    COALESCE(m.mortgage_avg_cpr, 0) AS mortgage_avg_cpr,
    COALESCE(m.mortgage_avg_wal_years, 0) AS mortgage_avg_wal_years,
    COALESCE(c.n_callable_bonds, 0) AS n_callable_bonds,
    COALESCE(c.callable_notional, 0) AS callable_notional,
    COALESCE(c.callable_straight_pv, 0) AS callable_straight_pv,
    COALESCE(c.callable_call_value, 0) AS callable_call_value,
    COALESCE(c.callable_callable_pv, 0) AS callable_callable_pv,
    COALESCE(c.callable_avg_integrated_vol, 0) AS callable_avg_integrated_vol
FROM {{ ref('stg_scenarios') }} s
LEFT JOIN mortgage_totals m ON m.scenario_id = s.id
LEFT JOIN callable_totals c ON c.scenario_id = s.id
ORDER BY s.id
