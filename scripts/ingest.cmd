@echo off
REM Regenerate synthetic Parquet data and reload into DuckDB.

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
python -m basel_ingestion.generate
if errorlevel 1 exit /b 1
python -m basel_ingestion.load
