
    
    

with all_values as (

    select
        shock_scenario as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_eve_bcbs368"
    group by shock_scenario

)

select *
from all_values
where value_field not in (
    'Parallel up','Parallel down','Short rate up','Short rate down','Steepener','Flattener'
)


