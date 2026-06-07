
    
    

with all_values as (

    select
        horizon_months as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_nii_horizon_stats"
    group by horizon_months

)

select *
from all_values
where value_field not in (
    '12','24','36'
)


