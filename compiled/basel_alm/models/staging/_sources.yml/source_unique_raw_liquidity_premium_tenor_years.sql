
    
    

select
    tenor_years as unique_field,
    count(*) as n_records

from "warehouse"."main"."liquidity_premium"
where tenor_years is not null
group by tenor_years
having count(*) > 1


