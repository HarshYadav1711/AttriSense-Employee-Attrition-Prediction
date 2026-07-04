@echo off
cd /d "%~dp0"
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1
if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\python.exe -m pip install --upgrade pip
    call .venv\Scripts\python.exe -m pip install -r requirements.txt
    call .venv\Scripts\python.exe -m pip install -e .
)
.venv\Scripts\python.exe -m streamlit run app/main.py %*
