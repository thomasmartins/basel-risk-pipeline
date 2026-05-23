{{ config(materialized='view') }}

SELECT
    id AS irrbb_id,
    date AS as_of_date,
    instrument,
    cashflow,
    maturity_date,
    tenor_bucket,
    pv01,
    rate_sensitivity,
    scenario_id
FROM {{ source('raw', 'irrbb') }}
