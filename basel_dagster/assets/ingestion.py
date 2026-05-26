"""Ingestion assets: Parquet generation + Parquet -> DuckDB load.

Asset graph:
    parquet_files -> raw/cashflows
                  -> raw/scenarios
                  -> raw/balance_sheet
                  -> raw/rwa
                  -> raw/irrbb
                  -> raw/params
The `raw/*` asset keys match what dbt's `source('raw', '<table>')` resolves
to by default in dagster-dbt, so the dbt models stitch onto these natively.
"""

from dagster import (
    AssetExecutionContext,
    AssetKey,
    AssetOut,
    MaterializeResult,
    MetadataValue,
    asset,
    multi_asset,
)

from basel_dagster.paths import DATA_RAW, DATA_WAREHOUSE
from basel_ingestion.generate import generate_all
from basel_ingestion.load import load_parquet_dir

RAW_TABLES: tuple[str, ...] = (
    "scenarios",
    "cashflows",
    "balance_sheet",
    "rwa",
    "irrbb",
    "params",
    "short_rate_history",
    "yield_curve",
    "liquidity_premium",
)


@asset(
    group_name="ingestion",
    compute_kind="python",
    description="Synthetic Parquet feed: writes 6 files under data/raw/.",
)
def parquet_files(context: AssetExecutionContext) -> dict[str, str]:
    written = generate_all(DATA_RAW)
    paths = {name: str(p) for name, p in written.items()}
    context.add_output_metadata(
        {
            "files": MetadataValue.json(paths),
            "n_tables": len(paths),
        }
    )
    return paths


@multi_asset(
    outs={
        table: AssetOut(
            key=AssetKey(["raw", table]),
            group_name="ingestion",
            description=f"DuckDB table loaded from data/raw/{table}.parquet.",
        )
        for table in RAW_TABLES
    },
    deps=[parquet_files],
    compute_kind="duckdb",
)
def raw_tables(context: AssetExecutionContext):
    counts = load_parquet_dir(DATA_RAW, DATA_WAREHOUSE)
    for table in RAW_TABLES:
        n = counts.get(table, 0)
        yield MaterializeResult(
            asset_key=AssetKey(["raw", table]),
            metadata={"row_count": MetadataValue.int(n)},
        )
