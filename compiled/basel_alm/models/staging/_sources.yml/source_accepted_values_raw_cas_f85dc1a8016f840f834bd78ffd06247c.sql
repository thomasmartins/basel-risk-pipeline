
    
    

with all_values as (

    select
        amortization_type as value_field,
        count(*) as n_records

    from "warehouse"."main"."cashflows"
    group by amortization_type

)

select *
from all_values
where value_field not in (
    'bullet','level'
)


