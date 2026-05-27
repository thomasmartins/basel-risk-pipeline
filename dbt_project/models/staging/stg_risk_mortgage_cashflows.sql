{{ config(materialized='view') }}

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
    pv_cpr - pv_scheduled AS pv_cpr_impact
FROM {{ source('risk', 'risk_mortgage_cashflows') }}
