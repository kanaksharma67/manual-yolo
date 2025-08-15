@echo off
echo Setting up Poker YOLO environment...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check OpenAI API key
echo.
echo Checking OpenAI API key...
if "%OPENAI_API_KEY%"=="" (
    echo ⚠ OPENAI_API_KEY not set!
    echo.
    echo To set your OpenAI API key, run one of these commands:
    echo   set OPENAI_API_KEY=your_key_here
    echo   setx OPENAI_API_KEY "your_key_here"
    echo.
    echo Or add it to your system environment variables.
    echo.
) else (
    echo ✅ OPENAI_API_KEY is set: %OPENAI_API_KEY:~0,10%...
)

echo.
echo Testing YOLO integration...
python test_yolo.py

echo.
echo Setup complete! To run the poker detector:
echo   python yolo.py
echo.
pause
