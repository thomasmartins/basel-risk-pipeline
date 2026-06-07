
    
    

select
    balance_sheet_id as unique_field,
    count(*) as n_records

from "warehouse"."main"."stg_balance_sheet"
where balance_sheet_id is not null
group by balance_sheet_id
having count(*) > 1


