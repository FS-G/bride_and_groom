@echo off
setlocal

REM ============================================================================
REM  Project Startup Script
REM  - Checks for a virtual environment and creates it if it doesn't exist.
REM  - Activates the virtual environment.
REM  - Installs/updates dependencies from requirements.txt.
REM  - Starts the FastAPI server and the main application.
REM ============================================================================

ECHO --- Checking for virtual environment...

REM Check if the 'venv' directory exists.
IF NOT EXIST ".\venv\" (
    ECHO Virtual environment not found. Creating it now...
    ECHO This might take a moment.
    
    REM Create the virtual environment. 
    REM 'py -m venv venv' is often more reliable on Windows if you have multiple Python versions.
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        ECHO ERROR: Failed to create the virtual environment. Please ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    ECHO Virtual environment created successfully.
) ELSE (
    ECHO Virtual environment found.
)

REM --- Activating the virtual environment ---
ECHO Activating the virtual environment...
call ".\venv\Scripts\activate"

REM --- Installing dependencies ---
ECHO Installing/updating dependencies from requirements.txt...
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO ERROR: Failed to install requirements. Check your requirements.txt file and network connection.
    pause
    exit /b 1
)
ECHO Dependencies are up to date.

REM --- Starting applications ---
ECHO Starting the FastAPI server in a new window...
start "FastAPI Server" cmd /k "uvicorn api:app --reload"

ECHO Starting the main application loop...
:loop
python app.py
ECHO Restarting app.py in 60 seconds...
timeout /t 60 > nul
goto loop