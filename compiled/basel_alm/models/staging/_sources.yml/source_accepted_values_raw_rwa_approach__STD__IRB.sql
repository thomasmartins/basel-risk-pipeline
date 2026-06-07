
    
    

with all_values as (

    select
        approach as value_field,
        count(*) as n_records

    from "warehouse"."main"."rwa"
    group by approach

)

select *
from all_values
where value_field not in (
    'STD','IRB'
)


