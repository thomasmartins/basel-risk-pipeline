

SELECT
    scenario_id,
    horizon_months,
    path_id,
    delta_nii,
    repricing_gap,
    avg_short_rate
FROM "warehouse"."main"."risk_nii_paths"