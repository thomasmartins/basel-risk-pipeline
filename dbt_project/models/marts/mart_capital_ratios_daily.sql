{{ config(materialized='table') }}

-- Daily capital ratios per scenario for the timeseries chart.

WITH rwa_d AS (
    SELECT as_of_date, scenario_id, SUM(rwa_amount) AS rwa
    FROM {{ ref('stg_rwa') }}
    GROUP BY as_of_date, scenario_id
)
SELECT
    COALESCE(r.as_of_date, b.as_of_date)   AS as_of_date,
    COALESCE(r.scenario_id, b.scenario_id) AS scenario_id,
    r.rwa,
    b.cet1,
    b.tier1,
    b.total_capital,
    b.cet1          / NULLIF(r.rwa, 0) AS cet1_ratio,
    b.tier1         / NULLIF(r.rwa, 0) AS tier1_ratio,
    b.total_capital / NULLIF(r.rwa, 0) AS total_capital_ratio
FROM rwa_d r
FULL OUTER JOIN {{ ref('int_balance_sheet_pivoted') }} b USING (as_of_date, scenario_id)
