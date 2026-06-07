
    
    

select
    tenor_years as unique_field,
    count(*) as n_records

from "warehouse"."main"."mart_ftp_curve"
where tenor_years is not null
group by tenor_years
having count(*) > 1


