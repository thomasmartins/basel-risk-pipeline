{{ config(materialized='view') }}

SELECT * FROM {{ source('risk', 'risk_model_metadata') }}
