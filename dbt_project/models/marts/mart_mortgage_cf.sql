{{ config(materialized='table') }}

-- Per-mortgage CPR-adjusted valuation summary, ordered by scenario then by
-- decreasing notional. The pv_cpr_impact column lets the dashboard rank
-- mortgages by how much prepayment shortened their PV vs the scheduled-only
-- counterfactual.

SELECT
    scenario_id,
    cashflow_id,
    notional,
    contract_rate,
    term_months,
    effective_term_months,
    avg_cpr,
    weighted_avg_life_years,
    pv_cpr,
    pv_scheduled,
    pv_cpr_impact
FROM {{ ref('stg_risk_mortgage_cashflows') }}
ORDER BY scenario_id, notional DESC
