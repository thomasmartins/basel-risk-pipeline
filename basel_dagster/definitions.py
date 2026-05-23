"""Top-level Dagster Definitions entrypoint.

`dagster dev -m basel_dagster.definitions` from the repo root with
PYTHONPATH=. picks this up.
"""

from __future__ import annotations

from dagster import Definitions

from basel_dagster.assets.ingestion import parquet_files, raw_tables
from basel_dagster.assets.risk_engine import risk_engine_run, risk_outputs_tables
from basel_dagster.assets.transformations import dbt_transformations
from basel_dagster.jobs import full_refresh_job
from basel_dagster.resources import dbt_resource
from basel_dagster.schedules import daily_full_refresh

defs = Definitions(
    assets=[parquet_files, raw_tables, dbt_transformations, risk_engine_run, risk_outputs_tables],
    jobs=[full_refresh_job],
    schedules=[daily_full_refresh],
    resources={"dbt": dbt_resource},
)
