
    
    

with all_values as (

    select
        item as value_field,
        count(*) as n_records

    from "warehouse"."main"."balance_sheet"
    group by item

)

select *
from all_values
where value_field not in (
    'CET1','Tier1','Total Capital','Total Assets','Total Liabilities'
)


