
    
    

with all_values as (

    select
        tenor_bucket as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_pv01_profile"
    group by tenor_bucket

)

select *
from all_values
where value_field not in (
    '0-1y','1-3y','3-5y','5-10y','10y+'
)


