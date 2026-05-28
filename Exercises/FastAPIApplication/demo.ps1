# Exercises/FastAPIApplication/demo.ps1 - Windows PowerShell end-to-end runner.
# Run from inside Exercises/FastAPIApplication/ with a clean working tree.
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv                                    # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Run the test suite -------------------------------------------------
# No running server needed - FastAPI's TestClient calls the app directly.
pytest -v

# --- 3. Smoke-launch the API ----------------------------------------------
# Background uvicorn on port 8000, give it a moment, curl two endpoints,
# then stop it. fastapi dev is the dev-time equivalent (with auto-reload).
Write-Host "Launching the API on http://127.0.0.1:8000 for a smoke check..."
$api = Start-Process -FilePath "uvicorn" `
    -ArgumentList "app.main:app", "--host", "127.0.0.1", "--port", "8000" `
    -NoNewWindow -PassThru
Start-Sleep -Seconds 4
# PowerShell aliases `curl` to Invoke-WebRequest, so call curl.exe explicitly.
curl.exe -s http://127.0.0.1:8000/; Write-Host ""
curl.exe -s http://127.0.0.1:8000/items/1; Write-Host ""
Stop-Process -Id $api.Id -Force -ErrorAction SilentlyContinue
Write-Host "API smoke check complete."
