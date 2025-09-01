@echo off
setlocal

REM ============================================================================
REM  Model Finetuning Script Runner - Drag & Drop Version
REM  - Checks for a Python virtual environment and creates it if it doesn't exist.
REM  - Activates the virtual environment.
REM  - Installs/updates dependencies from requirements.txt.
REM  - Runs the model fine-tuning script (finetune_drag_drop.py).
REM  - This version processes videos from finetune_drag_and_drop_folder instead of JSON config.
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

REM --- Running the fine-tuning script ---
ECHO.
ECHO ======================================================
ECHO  Starting the fine-tuning process (finetune_drag_drop.py)...
ECHO  Processing videos from finetune_drag_and_drop_folder...
ECHO ======================================================
ECHO.

python finetune_drag_drop.py
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO ******************************************************
    ECHO *  ERROR: The finetune_drag_drop.py script failed.   *
    ECHO *  Please check the output above for details.        *
    ECHO ******************************************************
) ELSE (
    ECHO.
    ECHO ******************************************************
    ECHO *  SUCCESS: The finetune_drag_drop.py script finished.*
    ECHO ******************************************************
)

ECHO.
ECHO Script execution complete. Press any key to close this window.
pause
endlocal
