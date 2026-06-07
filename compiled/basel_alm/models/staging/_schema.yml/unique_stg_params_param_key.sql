
    
    

select
    param_key as unique_field,
    count(*) as n_records

from "warehouse"."main"."stg_params"
where param_key is not null
group by param_key
having count(*) > 1


