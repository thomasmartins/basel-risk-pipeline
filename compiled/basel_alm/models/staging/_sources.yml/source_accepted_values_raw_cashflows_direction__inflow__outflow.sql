
    
    

with all_values as (

    select
        direction as value_field,
        count(*) as n_records

    from "warehouse"."main"."cashflows"
    group by direction

)

select *
from all_values
where value_field not in (
    'inflow','outflow'
)


