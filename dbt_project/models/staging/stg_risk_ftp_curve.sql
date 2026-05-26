{{ config(materialized='view') }}

SELECT * FROM {{ source('risk', 'risk_ftp_curve') }}
