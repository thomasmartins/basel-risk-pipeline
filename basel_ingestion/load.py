"""Load Parquet files in data/raw/ into the DuckDB warehouse.

Each Parquet file becomes a same-named table (CREATE OR REPLACE).
Idempotent — safe to re-run after regenerating data.

Run: `python -m basel_ingestion.load [--src data/raw] [--warehouse data/warehouse.duckdb]`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from basel_common.connection import warehouse_path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SRC = _REPO_ROOT / "data" / "raw"


def load_parquet_dir(src: Path, warehouse: Path) -> dict[str, int]:
    """Load every *.parquet under `src` as a table in `warehouse` (CREATE OR REPLACE).

    Returns {table_name: row_count}.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source dir not found: {src}")

    parquet_files = sorted(src.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files in {src}")

    warehouse.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(warehouse), read_only=False)
    try:
        counts: dict[str, int] = {}
        for pq in parquet_files:
            table = pq.stem
            con.execute(
                f"CREATE OR REPLACE TABLE {table} AS "
                f"SELECT * FROM read_parquet(?)",
                [str(pq)],
            )
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            counts[table] = n
            print(f"  loaded {table:15s} {n:>6d} rows")
        return counts
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=_DEFAULT_SRC)
    parser.add_argument("--warehouse", type=Path, default=None)
    args = parser.parse_args()

    target = args.warehouse if args.warehouse else warehouse_path()
    print(f"Loading {args.src.relative_to(_REPO_ROOT)} -> {target}")
    load_parquet_dir(args.src, target)


if __name__ == "__main__":
    main()
