
    
    

select
    id as unique_field,
    count(*) as n_records

from "warehouse"."main"."cashflows"
where id is not null
group by id
having count(*) > 1


