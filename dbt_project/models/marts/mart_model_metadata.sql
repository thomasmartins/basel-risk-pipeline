{{ config(materialized='table') }}

SELECT * FROM {{ ref('stg_risk_model_metadata') }}
