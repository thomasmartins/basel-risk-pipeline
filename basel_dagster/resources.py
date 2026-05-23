"""Dagster resources: dbt CLI + DuckDB project handle."""

from __future__ import annotations

import shutil
import site
import sys
from pathlib import Path

from dagster_dbt import DbtCliResource, DbtProject

from basel_dagster.paths import DBT_PROFILES_DIR, DBT_PROJECT_DIR


def _find_dbt_executable() -> str:
    """`pip install --user` puts dbt.exe under USER_BASE\\PythonXY\\Scripts, which
    is rarely on PATH. Probe the common locations so DbtCliResource validates
    on import without forcing the user to fix PATH first."""
    found = shutil.which("dbt")
    if found:
        return found
    py_ver = f"Python{sys.version_info.major}{sys.version_info.minor}"
    exe = "dbt.exe" if sys.platform == "win32" else "dbt"
    candidates = [
        Path(site.USER_BASE) / py_ver / "Scripts" / exe,
        Path(site.USER_BASE) / "Scripts" / exe,
        Path(sys.exec_prefix) / "Scripts" / exe,
        Path(sys.exec_prefix) / "bin" / exe,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return "dbt"  # let DbtCliResource raise with its native error message


dbt_project = DbtProject(
    project_dir=DBT_PROJECT_DIR,
    profiles_dir=DBT_PROFILES_DIR,
)
dbt_project.prepare_if_dev()

dbt_resource = DbtCliResource(
    project_dir=dbt_project,
    dbt_executable=_find_dbt_executable(),
)
