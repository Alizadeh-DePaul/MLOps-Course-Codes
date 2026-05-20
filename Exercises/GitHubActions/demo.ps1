# Exercises/GitHubActions/demo.ps1 - Windows PowerShell end-to-end runner.
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What it does:
#   1. Syncs a uv-managed venv from uv.lock.
#   2. Runs pytest locally so you see the same green/red CI will see.
#   3. Runs ruff and mypy locally (mirrors the codecheck workflow).
#   4. (Optional) Runs the GitHub workflows locally with `act` if available.
#
# Run it from inside this folder:
#   .\demo.ps1
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Write-Host "[1/4] Syncing venv from uv.lock ..."
uv sync --frozen --extra dev
. .\.venv\Scripts\Activate.ps1

# --- 2. Run the tests ------------------------------------------------------
Write-Host ""
Write-Host "[2/4] Running pytest ..."
uv run pytest -v

# --- 3. Code checks --------------------------------------------------------
Write-Host ""
Write-Host "[3/4] Running ruff + mypy ..."
uv run ruff check .
uv run ruff format --check .
uv run mypy simple_mlops

# --- 4. (Optional) Run workflows locally with act --------------------------
# `act` runs GitHub Actions workflows on your machine using Docker. Catch
# YAML/version issues here instead of burning CI minutes.
#   Install: winget install nektos.act
Write-Host ""
Write-Host "[4/4] (Optional) Local workflow validation with act ..."
if (Get-Command act -ErrorAction SilentlyContinue) {
    Write-Host "  act found - running .github/workflows/tests.yaml locally"
    act push --workflows .github\workflows\tests.yaml --container-architecture linux/amd64
} else {
    Write-Host "  act not installed - skipping local workflow run."
    Write-Host "  Install with: winget install nektos.act (Windows) | https://nektosact.com/"
}
