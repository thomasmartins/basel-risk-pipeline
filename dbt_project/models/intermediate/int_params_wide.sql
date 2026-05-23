{{ config(materialized='view') }}

-- One-row wide form of params for easy CROSS JOIN in downstream models.
-- Defaults match the Phase 0 seed values; if a param goes missing we fall
-- back to those rather than producing NULLs that propagate into ratios.

SELECT
    COALESCE(MAX(CASE WHEN param_key = 'haircut_level2a'           THEN param_value_numeric END), 0.15) AS haircut_level2a,
    COALESCE(MAX(CASE WHEN param_key = 'haircut_level2b'           THEN param_value_numeric END), 0.50) AS haircut_level2b,
    COALESCE(MAX(CASE WHEN param_key = 'lcr_inflow_cap'            THEN param_value_numeric END), 0.75) AS lcr_inflow_cap,
    COALESCE(MAX(CASE WHEN param_key = 'eve_tier1_breach_ratio'    THEN param_value_numeric END), 0.15) AS eve_tier1_breach_ratio,
    COALESCE(MAX(CASE WHEN param_key = 'capital_requirement_ratio' THEN param_value_numeric END), 0.08) AS capital_requirement_ratio
FROM {{ ref('stg_params') }}
