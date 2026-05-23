"""Dagster resources: dbt CLI + DuckDB project handle."""

from __future__ import annotations

import os
import shutil
import site
import sys
from pathlib import Path

from dagster_dbt import DbtCliResource, DbtProject

from basel_dagster.paths import DBT_PROFILES_DIR, DBT_PROJECT_DIR


def _ensure_dbt_on_path() -> str:
    """`pip install --user` puts dbt.exe under USER_BASE\\PythonXY\\Scripts, which
    is rarely on PATH. We probe the common locations, then prepend the dir to
    PATH at import time so every nested DbtCliResource construction (including
    the one that `DbtProject.prepare_if_dev()` makes internally without
    forwarding `dbt_executable`) can resolve `dbt`.
    """
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
            os.environ["PATH"] = str(c.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(c)
    return "dbt"  # let DbtCliResource raise with its native error message


_DBT_EXE = _ensure_dbt_on_path()

# profiles.yml uses `{{ env_var('BASEL_WAREHOUSE_PATH', ...) }}` for the
# DuckDB path. The fallback works only when cwd happens to be `dbt_project/`,
# so we set an absolute default here at import time. Doesn't override a
# user-set value.
from basel_dagster.paths import DATA_WAREHOUSE  # noqa: E402
os.environ.setdefault("BASEL_WAREHOUSE_PATH", str(DATA_WAREHOUSE.absolute()))


dbt_project = DbtProject(
    project_dir=DBT_PROJECT_DIR,
    profiles_dir=DBT_PROFILES_DIR,
)
dbt_project.prepare_if_dev()

dbt_resource = DbtCliResource(
    project_dir=dbt_project,
    dbt_executable=_DBT_EXE,
)
