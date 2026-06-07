
    
    

with all_values as (

    select
        product as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_nii_attribution_by_product"
    group by product

)

select *
from all_values
where value_field not in (
    'loan','deposit','bond'
)


