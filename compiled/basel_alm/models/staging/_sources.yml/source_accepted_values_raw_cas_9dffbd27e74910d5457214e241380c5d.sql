
    
    

with all_values as (

    select
        hqlatype as value_field,
        count(*) as n_records

    from "warehouse"."main"."cashflows"
    group by hqlatype

)

select *
from all_values
where value_field not in (
    'Level1','Level2A','Level2B','None'
)


