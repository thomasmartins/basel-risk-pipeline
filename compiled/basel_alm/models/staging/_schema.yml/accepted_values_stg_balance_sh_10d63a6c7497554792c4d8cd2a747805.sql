
    
    

with all_values as (

    select
        balance_sheet_item as value_field,
        count(*) as n_records

    from "warehouse"."main"."stg_balance_sheet"
    group by balance_sheet_item

)

select *
from all_values
where value_field not in (
    'CET1','Tier1','Total Capital','Total Assets','Total Liabilities'
)


