
    
    

with all_values as (

    select
        maturity_year_bucket as value_field,
        count(*) as n_records

    from "warehouse"."main"."int_cashflows_enriched"
    group by maturity_year_bucket

)

select *
from all_values
where value_field not in (
    '0-1y','1-3y','3-5y','5-10y','10y+'
)


