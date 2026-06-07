

-- ALMM-style survival horizon per (scenario, stress) pair.
-- survival_horizon_days is capped at the engine's max_horizon_days (365);
-- is_breached distinguishes a true breach from "still solvent at horizon".

SELECT
    scenario_id,
    stress_name,
    initial_cbc,
    survival_horizon_days,
    is_breached,
    peak_deficit,
    CASE
        WHEN NOT is_breached THEN 'survives horizon'
        WHEN survival_horizon_days <= 7   THEN 'critical (<= 1w)'
        WHEN survival_horizon_days <= 30  THEN 'severe (<= 1m)'
        WHEN survival_horizon_days <= 90  THEN 'moderate (<= 3m)'
        ELSE 'long (> 3m)'
    END AS severity_bucket
FROM "warehouse"."main"."stg_risk_survival_horizon"
ORDER BY scenario_id,
    CASE stress_name
        WHEN 'idiosyncratic' THEN 1
        WHEN 'market_wide'   THEN 2
        WHEN 'combined'      THEN 3
        ELSE 4 END