@echo off
REM Run the end-to-end risk-engine CLI, then reload its Parquet outputs into
REM DuckDB so dbt build sees the new schema immediately. Passes any extra args
REM through to argparse.
REM Example: scripts\risk_engine.cmd --model vasicek --n-paths 1000

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
python -m basel_risk_engine.run %*
if errorlevel 1 exit /b 1
python scripts\_reload_risk_outputs.py
