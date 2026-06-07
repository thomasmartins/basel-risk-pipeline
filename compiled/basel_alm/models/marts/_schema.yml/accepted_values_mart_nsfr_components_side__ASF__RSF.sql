
    
    

with all_values as (

    select
        side as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_nsfr_components"
    group by side

)

select *
from all_values
where value_field not in (
    'ASF','RSF'
)


