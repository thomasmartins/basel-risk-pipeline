

-- Per-scenario capital ratios: CET1 / Tier1 / Total Capital divided by total RWA.

WITH rwa_per_scn AS (
    SELECT scenario_id, SUM(rwa_amount) AS rwa
    FROM "warehouse"."main"."stg_rwa"
    GROUP BY scenario_id
),
bs_per_scn AS (
    SELECT
        scenario_id,
        SUM(cet1)               AS cet1,
        SUM(tier1)              AS tier1,
        SUM(total_capital)      AS total_capital,
        SUM(total_assets)       AS total_assets,
        SUM(total_liabilities)  AS total_liabilities
    FROM "warehouse"."main"."int_balance_sheet_pivoted"
    GROUP BY scenario_id
)
SELECT
    COALESCE(r.scenario_id, b.scenario_id) AS scenario_id,
    r.rwa,
    b.cet1,
    b.tier1,
    b.total_capital,
    b.total_assets,
    b.total_liabilities,
    b.cet1          / NULLIF(r.rwa, 0) AS cet1_ratio,
    b.tier1         / NULLIF(r.rwa, 0) AS tier1_ratio,
    b.total_capital / NULLIF(r.rwa, 0) AS total_capital_ratio
FROM rwa_per_scn r
FULL OUTER JOIN bs_per_scn b USING (scenario_id)