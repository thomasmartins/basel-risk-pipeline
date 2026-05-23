"""dbt transformations exposed as Dagster assets via @dbt_assets.

Every staging / intermediate / mart model in dbt_project/ becomes an asset
under the `transformations` group, with native dependencies on the
`raw/<table>` assets defined in `assets.ingestion`.
"""

from collections.abc import Iterable

from dagster import AssetExecutionContext
from dagster_dbt import DbtCliResource, dbt_assets

from basel_dagster.resources import dbt_project


@dbt_assets(manifest=dbt_project.manifest_path)
def dbt_transformations(context: AssetExecutionContext, dbt: DbtCliResource) -> Iterable:
    yield from dbt.cli(["build"], context=context).stream()
