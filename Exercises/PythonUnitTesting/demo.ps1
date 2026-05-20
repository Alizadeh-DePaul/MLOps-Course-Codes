# Exercises/PythonUnitTesting/demo.ps1 - Windows PowerShell end-to-end runner.
#
# Run AFTER you've filled in the TODOs in tests/*.py:
#   .\demo.ps1
#
# If Windows blocks execution the first time, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# Equivalent to demo.nu / demo.sh.

$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#
# Defaults to the CPU torch wheel (~200 MB). Override with:
#   $env:PYUT_EXTRA = "cuda"; .\demo.ps1
$Extra = if ($env:PYUT_EXTRA) { $env:PYUT_EXTRA } else { 'cpu' }
Write-Host "[1/4] Syncing venv with --extra $Extra ..."
uv sync "--extra=$Extra"
. .\.venv\Scripts\Activate.ps1

# --- 2. Run the tests ------------------------------------------------------
Write-Host ""
Write-Host "[2/4] Running pytest ..."
pytest -v

# --- 3. Run with coverage --------------------------------------------------
Write-Host ""
Write-Host "[3/4] Re-running under pytest-cov ..."
pytest --cov=models --cov=training --cov-report=term-missing --cov-report=html

# --- 4. Point at the HTML report -------------------------------------------
Write-Host ""
Write-Host "[4/4] HTML coverage report:"
Write-Host "  $((Get-Location).Path)\htmlcov\index.html"
Write-Host "  Open it in a browser to see line-by-line coverage."
