{{ config(materialized='view') }}

SELECT * FROM {{ source('risk', 'risk_nii_attribution_rows') }}
