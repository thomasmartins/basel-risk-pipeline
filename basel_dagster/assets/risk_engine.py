"""Risk engine assets: Vasicek calibration -> MC paths -> EVE/NII -> Parquet -> DuckDB.

Lineage:
    int_cashflows_enriched (dbt)  ┐
    mart_capital_ratios (dbt)     │
    raw/short_rate_history        ├─►  risk_engine_run  ─►  risk/<table> (multi-asset)
    raw/yield_curve               ┘
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
from duckdb import connect as duckdb_connect

from basel_dagster.paths import DATA_RISK_OUTPUTS, DATA_WAREHOUSE
from basel_risk_engine.run import run as run_risk_engine

RISK_TABLES: tuple[str, ...] = (
    "risk_eve_bcbs368",
    "risk_eve_supervisory",
    "risk_eve_distribution",
    "risk_nii_paths",
    "risk_rate_paths",
    "risk_model_metadata",
)


@asset(
    group_name="risk_engine",
    compute_kind="python",
    deps=[
        AssetKey(["int_cashflows_enriched"]),
        AssetKey(["mart_capital_ratios"]),
        AssetKey(["raw", "short_rate_history"]),
        AssetKey(["raw", "yield_curve"]),
    ],
    description="Calibrate Vasicek, simulate MC paths, value the book, write risk_outputs/*.parquet.",
)
def risk_engine_run(context: AssetExecutionContext) -> dict[str, str]:
    written = run_risk_engine(DATA_RISK_OUTPUTS, n_paths=2000, horizon_years=5.0, seed=7)
    paths = {name: str(p) for name, p in written.items()}
    context.add_output_metadata({"files": MetadataValue.json(paths)})
    return paths


@multi_asset(
    outs={
        table: AssetOut(
            key=AssetKey(["risk", table]),
            group_name="risk_engine",
            description=f"DuckDB table loaded from data/risk_outputs/{table}.parquet.",
        )
        for table in RISK_TABLES
    },
    deps=[risk_engine_run],
    compute_kind="duckdb",
)
def risk_outputs_tables(context: AssetExecutionContext):
    con = duckdb_connect(str(DATA_WAREHOUSE), read_only=False)
    try:
        for table in RISK_TABLES:
            pq = DATA_RISK_OUTPUTS / f"{table}.parquet"
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)",
                [str(pq)],
            )
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            yield MaterializeResult(
                asset_key=AssetKey(["risk", table]),
                metadata={"row_count": MetadataValue.int(n)},
            )
    finally:
        con.close()
