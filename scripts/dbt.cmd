@echo off
REM Run dbt from anywhere with the right profiles dir.
REM Usage: scripts\dbt.cmd build | scripts\dbt.cmd test | scripts\dbt.cmd docs generate | ...

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
python -c "from dbt.cli.main import dbtRunner; import sys; res = dbtRunner().invoke([*sys.argv[1:], '--project-dir', r'%CD%\dbt_project', '--profiles-dir', r'%CD%\dbt_project']); sys.exit(0 if res.success else 1)" %*
