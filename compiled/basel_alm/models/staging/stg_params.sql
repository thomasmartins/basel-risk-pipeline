

SELECT
    key AS param_key,
    value AS param_value,
    TRY_CAST(value AS DOUBLE) AS param_value_numeric
FROM "warehouse"."main"."params"