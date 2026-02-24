@echo off
setlocal enabledelayedexpansion
title Medical Document Analyzer - Starting...
color 0A

echo.
echo  ============================================================
echo   Medical Document Analyzer
echo   Starting up... please wait
echo  ============================================================
echo.

:: ----------------------------------------------------------------
:: Step 1: Check if Docker is installed
:: ----------------------------------------------------------------
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Docker Desktop is not installed.
    echo.
    echo  Opening the Docker Desktop download page...
    echo  Please install Docker Desktop, then run this script again.
    echo.
    start https://www.docker.com/products/docker-desktop/
    echo  After installing Docker Desktop:
    echo    1. Restart your computer
    echo    2. Double-click this file again
    echo.
    pause
    exit /b 1
)

echo  [OK] Docker is installed

:: ----------------------------------------------------------------
:: Step 2: Check if Docker daemon is running, start if needed
:: ----------------------------------------------------------------
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo  [..] Docker Desktop is not running. Starting it now...

    :: Try to start Docker Desktop
    if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" (
        start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
    ) else if exist "%LocalAppData%\Docker\Docker Desktop.exe" (
        start "" "%LocalAppData%\Docker\Docker Desktop.exe"
    ) else (
        echo  [!] Could not find Docker Desktop. Please start it manually.
        pause
        exit /b 1
    )

    echo  [..] Waiting for Docker to start (this can take 30-60 seconds)...

    :: Wait up to 2 minutes for Docker to be ready
    set /a attempts=0
    :wait_docker
    set /a attempts+=1
    if !attempts! gtr 24 (
        echo  [!] Docker did not start in time. Please start Docker Desktop manually and try again.
        pause
        exit /b 1
    )
    timeout /t 5 /nobreak >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        echo  [..] Still waiting... (!attempts!/24)
        goto wait_docker
    )
)

echo  [OK] Docker is running

:: ----------------------------------------------------------------
:: Step 3: Navigate to docker directory and start services
:: ----------------------------------------------------------------
cd /d "%~dp0"

echo.
echo  [..] Building and starting containers...
echo       (First time takes 5-10 minutes to download dependencies)
echo.

docker compose up --build -d
if %errorlevel% neq 0 (
    echo  [!] Failed to start containers. Check the error above.
    pause
    exit /b 1
)

echo  [OK] Containers started

:: ----------------------------------------------------------------
:: Step 4: Set up Ollama models (first time only)
:: ----------------------------------------------------------------
echo.
echo  [..] Checking AI models...

:: Check if MedGemma local model exists
docker exec medical-ollama ollama list 2>nul | findstr /i "medgemma-4b-local" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [..] Creating MedGemma model from local GGUF file...
    echo       (This takes 1-2 minutes, one time only)
    docker exec -w /models medical-ollama ollama create medgemma-4b-local -f Modelfile.medgemma
    if %errorlevel% neq 0 (
        echo  [!] Warning: MedGemma model creation failed. Extraction may not work.
        echo       Make sure the models/ folder contains medgemma-4b-it-Q4_K_M.gguf
    ) else (
        echo  [OK] MedGemma model ready
    )
) else (
    echo  [OK] MedGemma model already set up
)

:: Check if VLM model exists
docker exec medical-ollama ollama list 2>nul | findstr /i "minicpm" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [..] Downloading MiniCPM-V vision model (~5.5GB)...
    echo       (First time only — be patient)
    docker exec medical-ollama ollama pull minicpm-v
    if %errorlevel% neq 0 (
        echo  [!] Warning: VLM model download failed. Image extraction will use OCR only.
    ) else (
        echo  [OK] MiniCPM-V model ready
    )
) else (
    echo  [OK] MiniCPM-V model already downloaded
)

:: ----------------------------------------------------------------
:: Step 5: Wait for backend to be healthy
:: ----------------------------------------------------------------
echo.
echo  [..] Waiting for backend to be ready...

set /a attempts=0
:wait_backend
set /a attempts+=1
if !attempts! gtr 30 (
    echo  [!] Backend did not start in time. Check logs with:
    echo       docker compose logs backend
    pause
    exit /b 1
)
timeout /t 5 /nobreak >nul

:: Use curl via docker to check health (avoids needing curl on Windows)
docker exec medical-backend curl -sf http://localhost:8000/api/health >nul 2>&1
if %errorlevel% neq 0 (
    echo  [..] Backend starting... (!attempts!/30)
    goto wait_backend
)

echo  [OK] Backend is healthy

:: ----------------------------------------------------------------
:: Step 6: Open browser
:: ----------------------------------------------------------------
echo.
echo  ============================================================
echo   App is ready!
echo   Opening http://localhost:3000 in your browser...
echo.
echo   To stop the app, double-click stop.bat
echo   or close this window and run: docker compose down
echo  ============================================================
echo.

start http://localhost:3000

echo  Press any key to close this window (app keeps running)...
pause >nul
