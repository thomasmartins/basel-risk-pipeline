{{ config(materialized='view') }}

-- Cashflows enriched with maturity buckets, HQLA post-haircut, and signed amount.
-- Single source of truth for downstream LCR / NSFR / repricing-gap / heatmap marts.

WITH base AS (
    SELECT * FROM {{ ref('stg_cashflows') }}
),
params AS (
    SELECT * FROM {{ ref('int_params_wide') }}
),
with_days AS (
    SELECT
        b.*,
        DATE_DIFF('day', b.as_of_date, b.maturity_date) AS maturity_days
    FROM base b
)
SELECT
    w.cashflow_id,
    w.as_of_date,
    w.product,
    w.counterparty,
    w.maturity_date,
    w.maturity_days,
    w.reporting_bucket,
    w.amount,
    w.direction,
    w.hqla_type,
    w.asf_factor,
    w.rsf_factor,
    w.scenario_id,

    -- LCR-style day buckets for cashflow heatmap
    CASE
        WHEN w.maturity_days <=   7 THEN '0-7d'
        WHEN w.maturity_days <=  30 THEN '8-30d'
        WHEN w.maturity_days <=  90 THEN '31-90d'
        WHEN w.maturity_days <= 180 THEN '91-180d'
        WHEN w.maturity_days <= 365 THEN '181-365d'
        ELSE '>1y'
    END AS maturity_day_bucket,

    -- EBA-style year buckets for repricing gap / IRRBB
    CASE
        WHEN w.maturity_days <=   365 THEN '0-1y'
        WHEN w.maturity_days <=  3*365 THEN '1-3y'
        WHEN w.maturity_days <=  5*365 THEN '3-5y'
        WHEN w.maturity_days <= 10*365 THEN '5-10y'
        ELSE '10y+'
    END AS maturity_year_bucket,

    -- HQLA post-haircut: Level1 0, Level2A 15%, Level2B 50%, anything else excluded
    CASE
        WHEN w.hqla_type = 'Level1'  THEN w.amount
        WHEN w.hqla_type = 'Level2A' THEN w.amount * (1 - p.haircut_level2a)
        WHEN w.hqla_type = 'Level2B' THEN w.amount * (1 - p.haircut_level2b)
        ELSE 0
    END AS hqla_post_haircut,

    w.hqla_type IN ('Level1', 'Level2A', 'Level2B') AS is_hqla,

    -- Signed for cashflow-gap heatmaps (inflow +, outflow -)
    CASE WHEN w.direction = 'inflow' THEN w.amount ELSE -w.amount END AS signed_amount,

    -- NSFR contributions
    w.amount * w.asf_factor AS asf_contribution,
    w.amount * w.rsf_factor AS rsf_contribution
FROM with_days w
CROSS JOIN params p
