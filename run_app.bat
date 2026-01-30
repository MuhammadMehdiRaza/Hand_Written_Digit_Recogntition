@echo off
echo ========================================
echo   Digit Recognition App Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
)

REM Check if model exists
if not exist "models\digit_recognition_model.keras" (
    echo.
    echo ========================================
    echo   Model Not Found!
    echo ========================================
    echo.
    echo The trained model was not found.
    echo Would you like to train it now?
    echo This will take 5-10 minutes.
    echo.
    set /p train="Train now? (y/n): "
    if /i "%train%"=="y" (
        echo.
        echo Training model...
        python train_model.py
        echo.
    ) else (
        echo.
        echo Please train the model manually:
        echo   python train_model.py
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo   Starting Streamlit App...
echo ========================================
echo.
echo The app will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

streamlit run app.py
