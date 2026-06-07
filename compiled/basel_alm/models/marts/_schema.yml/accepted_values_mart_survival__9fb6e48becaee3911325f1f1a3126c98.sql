
    
    

with all_values as (

    select
        severity_bucket as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_survival_horizon"
    group by severity_bucket

)

select *
from all_values
where value_field not in (
    'survives horizon','critical (<= 1w)','severe (<= 1m)','moderate (<= 3m)','long (> 3m)'
)


