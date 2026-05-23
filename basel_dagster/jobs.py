"""Materialization jobs."""

from __future__ import annotations

from dagster import AssetSelection, define_asset_job

# Full pipeline: ingestion -> dbt transformations.
full_refresh_job = define_asset_job(
    name="full_refresh",
    selection=AssetSelection.all(),
    description="Regenerate Parquet, reload DuckDB, rebuild every dbt model.",
)
