# PowerShell script to set up Poker YOLO environment

Write-Host "Setting up Poker YOLO environment..." -ForegroundColor Green
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "Virtual environment created!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Check OpenAI API key
Write-Host ""
Write-Host "Checking OpenAI API key..." -ForegroundColor Yellow
$apiKey = $env:OPENAI_API_KEY
if (-not $apiKey) {
    Write-Host "⚠ OPENAI_API_KEY not set!" -ForegroundColor Red
    Write-Host ""
    Write-Host "To set your OpenAI API key, run one of these commands:" -ForegroundColor Yellow
    Write-Host "  `$env:OPENAI_API_KEY = 'your_key_here'" -ForegroundColor Cyan
    Write-Host "  [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'your_key_here', 'User')" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Or add it to your system environment variables." -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "✅ OPENAI_API_KEY is set: $($apiKey.Substring(0, [Math]::Min(10, $apiKey.Length)))..." -ForegroundColor Green
}

Write-Host ""
Write-Host "Testing YOLO integration..." -ForegroundColor Yellow
python test_yolo.py

Write-Host ""
Write-Host "Setup complete! To run the poker detector:" -ForegroundColor Green
Write-Host "  python yolo.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
