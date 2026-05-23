@echo off
REM Launches the Streamlit dashboard against the base anaconda env.
REM No editable install needed - PYTHONPATH covers `from basel_common ...`, `from src ...`, etc.

cd /d "%~dp0.."
call "C:\ProgramData\anaconda3\Scripts\activate.bat" "C:\ProgramData\anaconda3"
if errorlevel 1 (
    echo Failed to activate base anaconda environment.
    exit /b 1
)

set "PYTHONPATH=%CD%"
python -m streamlit run dashboard\Home.py %*
