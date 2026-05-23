{{ config(severity='warn') }}

-- Basic capital stack consistency: CET1 ≤ Tier1 ≤ Total Capital (per scenario).
-- Severity is `warn` because Phase 0's synthetic data picks each item
-- independently; Phase 2 (realistic balance sheets) is expected to honour
-- the ordering and this should be flipped back to `error` then.

SELECT
    scenario_id,
    cet1,
    tier1,
    total_capital
FROM {{ ref('mart_capital_ratios') }}
WHERE cet1 > tier1
   OR tier1 > total_capital
