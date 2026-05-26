{{ config(materialized='view') }}

SELECT
    valuation_date,
    tenor_label,
    tenor_years,
    lp_bps
FROM {{ source('raw', 'liquidity_premium') }}
