
    
    

select
    rwa_id as unique_field,
    count(*) as n_records

from "warehouse"."main"."stg_rwa"
where rwa_id is not null
group by rwa_id
having count(*) > 1


