

-- Percentile summary of MC ΔNII per (scenario, horizon).

SELECT
    scenario_id,
    horizon_months,
    COUNT(*)                                AS n_paths,
    AVG(delta_nii)                          AS mean,
    STDDEV_POP(delta_nii)                   AS stddev,
    MIN(delta_nii)                          AS min,
    QUANTILE_CONT(delta_nii, 0.05)          AS p05,
    QUANTILE_CONT(delta_nii, 0.50)          AS median,
    QUANTILE_CONT(delta_nii, 0.95)          AS p95,
    MAX(delta_nii)                          AS max,
    ANY_VALUE(repricing_gap)                AS repricing_gap
FROM "warehouse"."main"."stg_risk_nii_paths"
GROUP BY scenario_id, horizon_months