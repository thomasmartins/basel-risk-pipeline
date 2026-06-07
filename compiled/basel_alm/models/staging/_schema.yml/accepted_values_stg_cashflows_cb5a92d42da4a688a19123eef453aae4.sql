
    
    

with all_values as (

    select
        hqla_type as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_cashflows"
    group by hqla_type

)

select *
from all_values
where value_field not in (
    'Level1','Level2A','Level2B','None'
)


