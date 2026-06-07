
    
    

select
    id as unique_field,
    count(*) as n_records

from "warehouse"."main"."balance_sheet"
where id is not null
group by id
having count(*) > 1


