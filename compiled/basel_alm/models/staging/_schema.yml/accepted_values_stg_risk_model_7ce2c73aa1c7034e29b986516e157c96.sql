
    
    

with all_values as (

    select
        model_family as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_risk_model_metadata"
    group by model_family

)

select *
from all_values
where value_field not in (
    'hull_white_1f','vasicek_1f'
)


