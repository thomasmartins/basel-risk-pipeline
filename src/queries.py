"""DuckDB-backed read layer for the dashboard.

Phase 0: thin SELECTs against the warehouse populated by `basel_ingestion`.
Phase 1+: most callers will move to dbt marts and these helpers will shrink
to scenario / params lookups.

Public function signatures are preserved from the SQLAlchemy era so the
dashboard pages and `compute.py` don't need to change here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from basel_common.connection import duckdb_connect


@st.cache_resource(show_spinner=False)
def get_connection():
    """Single read-only DuckDB connection per Streamlit session."""
    return duckdb_connect(read_only=True)


def _query(sql: str, params: list | None = None) -> pd.DataFrame:
    return get_connection().execute(sql, params or []).fetchdf()


@st.cache_data(show_spinner=False)
def get_cashflows(start_date=None, end_date=None, scenario_id=None) -> pd.DataFrame:
    sql = "SELECT * FROM cashflows WHERE 1=1"
    params: list = []
    if start_date is not None:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date is not None:
        sql += " AND date <= ?"
        params.append(end_date)
    if scenario_id is not None:
        sql += " AND scenario_id = ?"
        params.append(scenario_id)
    return _query(sql, params)


@st.cache_data(show_spinner=False)
def get_rwa(start_date=None, end_date=None, scenario_id=None) -> pd.DataFrame:
    sql = "SELECT * FROM rwa WHERE 1=1"
    params: list = []
    if start_date is not None:
        sql += " AND date >= ?"
        params.append(start_date)
    if end_date is not None:
        sql += " AND date <= ?"
        params.append(end_date)
    if scenario_id is not None:
        sql += " AND scenario_id = ?"
        params.append(scenario_id)
    return _query(sql, params)


@st.cache_data(show_spinner=False)
def get_irrbb(scenario_id=None) -> pd.DataFrame:
    sql = "SELECT * FROM irrbb"
    params: list = []
    if scenario_id is not None:
        sql += " WHERE scenario_id = ?"
        params.append(scenario_id)
    return _query(sql, params)


@st.cache_data(show_spinner=False)
def get_balance_sheet(scenario_id=None) -> pd.DataFrame:
    sql = "SELECT * FROM balance_sheet"
    params: list = []
    if scenario_id is not None:
        sql += " WHERE scenario_id = ?"
        params.append(scenario_id)
    return _query(sql, params)


@st.cache_data(show_spinner=False)
def get_scenarios() -> pd.DataFrame:
    return _query("SELECT * FROM scenarios ORDER BY id")


@st.cache_data(show_spinner=False)
def get_params() -> dict:
    df = _query("SELECT key, value FROM params")
    return dict(zip(df["key"], df["value"]))


@st.cache_data(show_spinner=False)
def get_mart(name: str, scenario_id=None) -> pd.DataFrame:
    """Read a dbt mart, optionally filtered by scenario_id.

    Scenario filter is applied only if the mart has a `scenario_id` column.
    """
    has_scenario = (
        _query("SELECT 1 FROM information_schema.columns "
               "WHERE table_name = ? AND column_name = 'scenario_id'", [name]).shape[0] > 0
    )
    if scenario_id is not None and has_scenario:
        return _query(f"SELECT * FROM {name} WHERE scenario_id = ?", [scenario_id])
    return _query(f"SELECT * FROM {name}")


if __name__ == "__main__":
    # Smoke test (outside Streamlit; cache decorators are no-ops with a warning).
    print(get_scenarios())
    print(get_params())
    print(get_cashflows(scenario_id=1).head())
