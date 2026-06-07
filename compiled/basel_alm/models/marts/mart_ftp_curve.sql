

-- Internal FTP curve = base wholesale yield + per-tenor liquidity premium add-on.

SELECT
    tenor_years,
    base_yield,
    lp_bps,
    ftp_yield
FROM "warehouse"."main"."stg_risk_ftp_curve"
ORDER BY tenor_years