@echo off
REM Start the Dagster web UI for local development.
REM Visit http://localhost:3000 once it logs "Serving dagster-webserver on ...".

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
set "BASEL_WAREHOUSE_PATH=%CD%\data\warehouse.duckdb"
if not defined DAGSTER_HOME set "DAGSTER_HOME=%CD%\.dagster_home"
if not exist "%DAGSTER_HOME%" mkdir "%DAGSTER_HOME%"

python -m dagster dev -m basel_dagster.definitions %*
