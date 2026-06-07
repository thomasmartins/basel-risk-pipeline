
    
    

select
    scenario_id as unique_field,
    count(*) as n_records

from "warehouse"."main"."mart_eve_supervisory"
where scenario_id is not null
group by scenario_id
having count(*) > 1


