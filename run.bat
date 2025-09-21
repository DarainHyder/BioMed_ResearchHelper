@echo off
:: Biomedical Research Assistant - Windows Batch Runner
:: This script helps you run the system easily on Windows

setlocal

echo =====================================
echo   Biomedical Research Assistant
echo =====================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv\" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
    echo.
)

:: Activate virtual environment
call venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Check if requirements are installed
if not exist "venv\Lib\site-packages\fastapi" (
    echo Installing requirements...
    echo This may take 10-15 minutes on first run...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
    echo Requirements installed successfully.
    echo.
)

:: Check configuration
if not exist ".env" (
    echo Configuration file not found.
    if exist ".env.template" (
        copy .env.template .env >nul
        echo .env file created from template.
        echo IMPORTANT: Please edit .env file and set your email address!
        echo.
        echo Opening .env file in notepad...
        start notepad .env
        echo After editing .env, press any key to continue...
        pause >nul
    ) else (
        echo ERROR: .env.template not found
        pause
        exit /b 1
    )
)

:: Main menu
:menu
echo.
echo What would you like to do?
echo.
echo 1. Check configuration
echo 2. Setup system (first time setup - may take 30-60 minutes)
echo 3. Start API server
echo 4. Start web dashboard
echo 5. Run search demo
echo 6. Run summarization demo
echo 7. Run topic analysis demo
echo 8. Start both server and dashboard
echo 9. Exit
echo.
set /p choice="Enter your choice (1-9): "

if "%choice%"=="1" goto check
if "%choice%"=="2" goto setup
if "%choice%"=="3" goto server
if "%choice%"=="4" goto dashboard
if "%choice%"=="5" goto demo_search
if "%choice%"=="6" goto demo_summary
if "%choice%"=="7" goto demo_topics
if "%choice%"=="8" goto both
if "%choice%"=="9" goto exit
goto menu

:check
echo.
echo Checking configuration...
python main.py check
if errorlevel 1 (
    echo Configuration check failed. Please check your .env file.
) else (
    echo Configuration check passed!
)
echo.
pause
goto menu

:setup
echo.
echo Starting system setup...
echo This will:
echo - Download research papers from PubMed
echo - Process and analyze the data
echo - Create AI embeddings and indexes
echo - Set up topic models
echo.
echo This may take 30-60 minutes depending on your settings.
set /p confirm="Continue? (y/n): "
if /i not "%confirm%"=="y" goto menu

echo.
echo Starting setup...
python main.py setup
if errorlevel 1 (
    echo Setup failed. Please check the error messages above.
) else (
    echo Setup completed successfully!
    echo You can now start the server and dashboard.
)
echo.
pause
goto menu

:server
echo.
echo Starting API server...
echo API will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo.
python main.py server
goto menu

:dashboard
echo.
echo Starting web dashboard...
echo Dashboard will be available at: http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.
python main.py dashboard
goto menu

:demo_search
echo.
echo Running search demonstration...
python main.py demo-search
echo.
pause
goto menu

:demo_summary
echo.
echo Running summarization demonstration...
python main.py demo-summary
echo.
pause
goto menu

:demo_topics
echo.
echo Running topic analysis demonstration...
python main.py demo-topics
echo.
pause
goto menu

:both
echo.
echo Starting both API server and web dashboard...
echo.
echo Starting API server in background...
start /b python main.py server

:: Wait a moment for server to start
timeout /t 5 /nobreak >nul

echo Starting web dashboard...
echo Dashboard will be available at: http://localhost:8501
echo API will be available at: http://localhost:8000
echo.
echo Press Ctrl+C to stop
python main.py dashboard

:: Clean up background process
taskkill /f /im python.exe /fi "WINDOWTITLE eq *main.py server*" >nul 2>&1
goto menu

:exit
echo.
echo Thanks for using Biomedical Research Assistant!
echo.
pause
exit /b 0