"""Path constants for the Dagster code location.

Centralised so the dbt project, warehouse, and raw data dirs are referenced
consistently across assets and resources.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DBT_PROJECT_DIR = REPO_ROOT / "dbt_project"
DBT_PROFILES_DIR = DBT_PROJECT_DIR  # profiles.yml lives next to dbt_project.yml
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_RISK_OUTPUTS = REPO_ROOT / "data" / "risk_outputs"
DATA_WAREHOUSE = REPO_ROOT / "data" / "warehouse.duckdb"
