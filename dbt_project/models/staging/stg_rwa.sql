{{ config(materialized='view') }}

SELECT
    id AS rwa_id,
    date AS as_of_date,
    exposure_id,
    asset_class,
    approach,
    amount AS exposure_amount,
    risk_weight,
    rwa_amount,
    capital_requirement,
    scenario_id
FROM {{ source('raw', 'rwa') }}
