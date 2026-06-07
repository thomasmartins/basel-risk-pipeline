
    
    

select
    key as unique_field,
    count(*) as n_records

from "warehouse"."main"."params"
where key is not null
group by key
having count(*) > 1


