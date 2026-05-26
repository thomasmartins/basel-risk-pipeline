@echo off
REM Run the end-to-end risk-engine CLI. Passes any args through to argparse.
REM Example: scripts\risk_engine.cmd --model vasicek --n-paths 1000

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
python -m basel_risk_engine.run %*
