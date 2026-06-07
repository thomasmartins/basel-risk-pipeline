

with meet_condition as(
  select *
  from "warehouse"."main"."stg_risk_callable_bonds"
),

validation_errors as (
  select *
  from meet_condition
  where
    -- never true, defaults to an empty result set. Exists to ensure any combo of the `or` clauses below succeeds
    1 = 2
    -- records with a value >= min_value are permitted. The `not` flips this to find records that don't meet the rule.
    or not call_value >= 0
)

select *
from validation_errors

