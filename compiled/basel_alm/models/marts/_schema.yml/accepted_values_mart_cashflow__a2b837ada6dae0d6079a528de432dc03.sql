
    
    

with all_values as (

    select
        maturity_day_bucket as value_field,
        count(*) as n_records

    from "warehouse"."main"."mart_cashflow_gap"
    group by maturity_day_bucket

)

select *
from all_values
where value_field not in (
    '0-7d','8-30d','31-90d','91-180d','181-365d','>1y'
)


