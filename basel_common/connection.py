"""DuckDB connection helper.

Resolution order for the warehouse path:
1. Streamlit secrets `[duckdb] path`
2. Env var `BASEL_WAREHOUSE_PATH`
3. Default: `<repo_root>/data/warehouse.duckdb`

The repo root is taken as the parent of the `basel_common` package directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = _REPO_ROOT / "data" / "warehouse.duckdb"


def warehouse_path() -> Path:
    try:
        import streamlit as st

        cfg = st.secrets.get("duckdb")
        if cfg and cfg.get("path"):
            p = Path(cfg["path"])
            return p if p.is_absolute() else (_REPO_ROOT / p)
    except Exception:
        pass

    env = os.environ.get("BASEL_WAREHOUSE_PATH")
    if env:
        return Path(env)

    return _DEFAULT_PATH


def duckdb_connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    path = warehouse_path()
    if read_only and not path.exists():
        raise FileNotFoundError(
            f"DuckDB warehouse not found at {path}. "
            "Run the ingestion pipeline first: "
            "`python -m basel_ingestion.generate` then `python -m basel_ingestion.load`."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)
