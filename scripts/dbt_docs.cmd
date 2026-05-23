@echo off
REM Generate dbt docs and serve them locally on http://localhost:8081.
REM Press Ctrl+C to stop.

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
set "BASEL_WAREHOUSE_PATH=%CD%\data\warehouse.duckdb"
python -c "from dbt.cli.main import dbtRunner; import sys; res = dbtRunner().invoke(['docs','generate','--project-dir', r'%CD%\dbt_project','--profiles-dir', r'%CD%\dbt_project']); sys.exit(0 if res.success else 1)"
if errorlevel 1 (
    echo dbt docs generate failed.
    exit /b 1
)
python -c "from dbt.cli.main import dbtRunner; r=dbtRunner(); r.invoke(['docs','serve','--port','8081','--project-dir', r'%CD%\dbt_project','--profiles-dir', r'%CD%\dbt_project','--no-browser'])"
