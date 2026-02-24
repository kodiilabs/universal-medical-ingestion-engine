@echo off
title Medical Document Analyzer - Stopping...
color 0C

echo.
echo  ============================================================
echo   Medical Document Analyzer
echo   Stopping all services...
echo  ============================================================
echo.

cd /d "%~dp0"
docker compose down

echo.
echo  [OK] All services stopped.
echo.
echo  Your uploaded documents and AI models are preserved.
echo  Double-click start.bat to start again.
echo.
pause
