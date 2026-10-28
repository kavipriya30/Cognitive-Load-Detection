@echo off
title Cognitive Load Detection AI Launcher
echo ======================================================================
echo    COGNITIVE LOAD DETECTION USING TYPING PATTERNS
echo    Machine Learning & Keystroke Biometrics Platform
echo ======================================================================
echo.
echo [1/2] Starting Modern Website Server on http://localhost:8000 ...
start "" python -m uvicorn server:app --host 0.0.0.0 --port 8000

echo [2/2] Starting Streamlit Dashboard on http://localhost:8501 ...
start "" python -m streamlit run app/app.py --server.port 8501

echo.
echo ======================================================================
echo Both interfaces are launching:
echo    - Modern Web Application:  http://localhost:8000
echo    - Streamlit Dashboard:     http://localhost:8501
echo ======================================================================
echo.
timeout /t 3 >nul
start http://localhost:8000
pause
