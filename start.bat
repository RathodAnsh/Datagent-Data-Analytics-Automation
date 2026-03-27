@echo off
echo ============================================
echo   DATAGENT - Starting All Services
echo ============================================

set BASE=%~dp0

echo [1/2] Syncing .env to all backends...
copy /Y "%BASE%.env" "%BASE%backends\main\.env" >nul
copy /Y "%BASE%.env" "%BASE%backends\excel\.env" >nul
copy /Y "%BASE%.env" "%BASE%backends\python_analysis\.env" >nul
copy /Y "%BASE%.env" "%BASE%backends\r_analysis\.env" >nul
copy /Y "%BASE%.env" "%BASE%backends\sql_analysis\.env" >nul
echo Done.

echo [2/2] Starting all services...

:: ── BACKENDS ──
start "MAIN-BACKEND-8000" cmd /k "cd /d "%BASE%backends\main" && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000"
start "EXCEL-BACKEND-8001" cmd /k "cd /d "%BASE%backends\excel" && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8001"
start "PYTHON-BACKEND-8002" cmd /k "cd /d "%BASE%backends\python_analysis" && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8002"
start "R-BACKEND-8003" cmd /k "cd /d "%BASE%backends\r_analysis" && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8003"
start "SQL-BACKEND-8004" cmd /k "cd /d "%BASE%backends\sql_analysis" && venv\Scripts\python.exe -m uvicorn main:app --reload --port 8004"

:: ── FRONTENDS ──
start "AUTH-FRONTEND-3000" cmd /k "cd /d "%BASE%frontends\auth" && set NODE_OPTIONS=--max-old-space-size=4096 && node_modules\.bin\next.cmd dev"
start "PIPELINE-FRONTEND-5173" cmd /k "cd /d "%BASE%frontends\pipeline" && npm run dev"
start "PYTHON-FRONTEND-3001" cmd /k "cd /d "%BASE%frontends\python_analysis" && npm run dev"
start "R-FRONTEND-3002" cmd /k "cd /d "%BASE%frontends\r_analysis" && npm run dev"
start "SQL-FRONTEND-3003" cmd /k "cd /d "%BASE%frontends\sql_analysis" && npm run dev"

echo.
echo ============================================
echo   All 10 services started!
echo   Wait ~30 seconds then open:
echo   http://localhost:3000  (Login)
echo   http://localhost:5173  (Pipeline)
echo ============================================
pause