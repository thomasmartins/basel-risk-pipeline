{{ config(materialized='table') }}

-- Percentile summary of the MC ΔEVE distribution per scenario.

SELECT
    scenario_id,
    COUNT(*)                                     AS n_paths,
    AVG(delta_eve)                               AS mean,
    STDDEV_POP(delta_eve)                        AS stddev,
    MIN(delta_eve)                               AS min,
    QUANTILE_CONT(delta_eve, 0.01)               AS p01,
    QUANTILE_CONT(delta_eve, 0.05)               AS p05,
    QUANTILE_CONT(delta_eve, 0.50)               AS median,
    QUANTILE_CONT(delta_eve, 0.95)               AS p95,
    QUANTILE_CONT(delta_eve, 0.99)               AS p99,
    MAX(delta_eve)                               AS max,
    QUANTILE_CONT(ABS(delta_eve), 0.99)          AS abs_p99
FROM {{ ref('stg_risk_eve_distribution') }}
GROUP BY scenario_id
