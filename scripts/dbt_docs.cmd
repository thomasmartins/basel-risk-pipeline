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
python -c "from dbt.cli.main import dbtRunner; r=dbtRunner(); r.invoke(['docs','generate','--project-dir', r'%CD%\dbt_project','--profiles-dir', r'%CD%\dbt_project'])"
python -c "from dbt.cli.main import dbtRunner; r=dbtRunner(); r.invoke(['docs','serve','--port','8081','--project-dir', r'%CD%\dbt_project','--profiles-dir', r'%CD%\dbt_project','--no-browser'])"
