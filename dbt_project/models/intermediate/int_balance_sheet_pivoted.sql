{{ config(materialized='view') }}

-- One row per (as_of_date, scenario_id) with each balance-sheet item as a column.
-- Sums across multiple raw rows per (date, item, scenario_id) — the generator
-- emits more than one such row.

SELECT
    as_of_date,
    scenario_id,
    SUM(CASE WHEN balance_sheet_item = 'CET1'              THEN amount ELSE 0 END) AS cet1,
    SUM(CASE WHEN balance_sheet_item = 'Tier1'             THEN amount ELSE 0 END) AS tier1,
    SUM(CASE WHEN balance_sheet_item = 'Total Capital'     THEN amount ELSE 0 END) AS total_capital,
    SUM(CASE WHEN balance_sheet_item = 'Total Assets'      THEN amount ELSE 0 END) AS total_assets,
    SUM(CASE WHEN balance_sheet_item = 'Total Liabilities' THEN amount ELSE 0 END) AS total_liabilities
FROM {{ ref('stg_balance_sheet') }}
GROUP BY as_of_date, scenario_id
