

-- RWA by (scenario, approach, asset_class). Drives the treemap and STD/IRB rollups.

SELECT
    scenario_id,
    approach,
    asset_class,
    SUM(exposure_amount) AS exposure_amount,
    SUM(rwa_amount)      AS rwa_amount,
    SUM(capital_requirement) AS capital_requirement
FROM "warehouse"."main"."stg_rwa"
GROUP BY scenario_id, approach, asset_class