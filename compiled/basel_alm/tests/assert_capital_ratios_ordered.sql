-- Basic capital stack consistency: CET1 ≤ Tier1 ≤ Total Capital (per scenario).
-- The Phase 0 generator picked each item independently and breached this; the
-- Phase 3 polish rewrote generate_balance_sheet so the stack is enforced by
-- construction (Tier1 = CET1 + AT1, Total = Tier1 + Tier2 with positive
-- increments), so severity is back to the default `error`.

SELECT
    scenario_id,
    cet1,
    tier1,
    total_capital
FROM "warehouse"."main"."mart_capital_ratios"
WHERE cet1 > tier1
   OR tier1 > total_capital