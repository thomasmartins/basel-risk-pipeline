"""Helper: CREATE OR REPLACE TABLE for each Parquet in data/risk_outputs/.

Needed when scripts/risk_engine.cmd is run standalone (not via Dagster) — the
engine writes Parquets but does not load them into DuckDB. dbt then sees stale
schema unless we run this script in between.
"""

from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = REPO_ROOT / "data" / "warehouse.duckdb"
RISK_OUTPUTS = REPO_ROOT / "data" / "risk_outputs"

con = duckdb.connect(str(WAREHOUSE))
try:
    for p in sorted(RISK_OUTPUTS.glob("*.parquet")):
        name = p.stem
        con.execute(
            f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet(?)",
            [str(p)],
        )
        n = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"  reloaded {name:30s} {n:>7d} rows")
finally:
    con.close()
