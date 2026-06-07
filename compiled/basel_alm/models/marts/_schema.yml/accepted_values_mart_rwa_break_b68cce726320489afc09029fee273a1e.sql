
    
    

with all_values as (

    select
        asset_class as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_rwa_breakdown"
    group by asset_class

)

select *
from all_values
where value_field not in (
    'mortgage','corporate','sovereign','retail'
)


