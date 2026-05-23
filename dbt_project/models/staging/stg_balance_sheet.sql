{{ config(materialized='view') }}

SELECT
    id AS balance_sheet_id,
    date AS as_of_date,
    item AS balance_sheet_item,
    amount,
    scenario_id
FROM {{ source('raw', 'balance_sheet') }}
