# Exercises/ContinuousMachineLearning/demo.ps1 - Windows PowerShell end-to-end runner.
#
# What it does:
#   1. Syncs a uv-managed venv and installs the project deps.
#   2. Trains the model (writes classification_report.txt + confusion_matrix.png).
#   3. (Optional) Validates the workflow YAML locally with `act` if installed.
#
# What it does NOT do:
#   - Post a PR comment. That only happens when GitHub Actions runs the
#     workflow on a real `git push` (you cannot post to a PR from your laptop).
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Write-Host "[1/3] Creating venv and installing deps with uv ..."
uv venv                                    # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e .                        # alt: pip install -e .

# --- 2. Train the model ----------------------------------------------------
Write-Host ""
Write-Host "[2/3] Training model ..."
python simple_mlops\train_model_cml.py

Write-Host ""
Write-Host "      Artifacts produced:"
Get-ChildItem classification_report.txt, confusion_matrix.png

# --- 3. (Optional) Validate the workflow locally with act ------------------
# `act` runs GitHub Actions workflows on your machine using Docker.
Write-Host ""
Write-Host "[3/3] (Optional) Local workflow validation with act ..."
if (Get-Command act -ErrorAction SilentlyContinue) {
    Write-Host "  act found - running .github/workflows/cml.yaml locally"
    act push --workflows .github/workflows/cml.yaml --container-architecture linux/amd64
} else {
    Write-Host "  act not installed - skipping local workflow run."
    Write-Host "  Install with: winget install nektos.act"
}
