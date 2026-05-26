{{ config(materialized='view') }}

SELECT
    id AS cashflow_id,
    date AS as_of_date,
    product,
    counterparty,
    maturity_date,
    bucket AS reporting_bucket,
    amount,
    direction,
    hqlatype AS hqla_type,
    asf_factor,
    rsf_factor,
    customer_rate,
    scenario_id
FROM {{ source('raw', 'cashflows') }}
