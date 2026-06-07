
    
    

with child as (
    select scenario_id as from_field
    from "warehouse"."main"."cashflows"
    where scenario_id is not null
),

parent as (
    select id as to_field
    from "warehouse"."main"."scenarios"
)

select
    from_field

from child
left join parent
    on child.from_field = parent.to_field

where parent.to_field is null


