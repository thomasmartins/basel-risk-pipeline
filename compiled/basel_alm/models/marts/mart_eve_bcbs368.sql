

-- Pass-through: deterministic ΔEVE under the six BCBS 368 prescribed scenarios.

SELECT
    scenario_id,
    shock_scenario,
    delta_eve
FROM "warehouse"."main"."stg_risk_eve_bcbs368"