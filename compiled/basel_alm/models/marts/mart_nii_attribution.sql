

-- Book-level NII decomposition per scenario under matched-funded FTP:
--   customer_margin  = sign * notional * (customer_rate - ftp_behavioural)
--   funding_margin   = sign * notional * (ftp_behavioural - r_overnight)
--   behavioral_value = sign * notional * (ftp_contractual - ftp_behavioural)   for NMDs only
--   nii_total        = customer_margin + funding_margin
-- behavioral_value is a sub-component of customer_margin (the slice attributable
-- to pricing NMDs at behavioural rather than contractual maturity).

SELECT
    scenario_id,
    customer_margin,
    funding_margin,
    behavioral_value,
    nii_total
FROM "warehouse"."main"."stg_risk_nii_attribution"
ORDER BY scenario_id