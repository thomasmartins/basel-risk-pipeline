
    
    

with all_values as (

    select
        stress_name as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_survival_horizon"
    group by stress_name

)

select *
from all_values
where value_field not in (
    'idiosyncratic','market_wide','combined'
)


