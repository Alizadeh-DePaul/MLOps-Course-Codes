# Exercises/WandB/demo.ps1 - Windows PowerShell end-to-end runner for the WandB exercise.
# Run from inside Exercises/WandB/ after `wandb login` (or with WANDB_API_KEY set).
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What this does (mirrors the exercise page steps 2-6):
#   1. Creates a venv with uv and installs the package
#   2. Confirms WandB credentials are available
#   3. Runs train.py (basic scalar logging)
#   4. Runs train_advanced.py (images / histograms / matplotlib / confusion matrix)
#
# This script does NOT run `wandb agent` because sweep agents block forever.
# See the README for the manual sweep flow.
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv                                    # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e .                        # alt: pip install -e .

# --- 2. Credential check ---------------------------------------------------
# Bail out early if neither WANDB_API_KEY nor %USERPROFILE%\.netrc is set up.
$netrc = Join-Path $env:USERPROFILE ".netrc"
if (-not $env:WANDB_API_KEY -and -not (Test-Path $netrc)) {
    Write-Error "No W&B credentials found. Run ``wandb login`` first, or set WANDB_API_KEY."
    exit 1
}

# --- 3. Basic run ----------------------------------------------------------
# Logs `loss` as a scalar every 100 mini-batches.
python train.py

# --- 4. Advanced run -------------------------------------------------------
# Adds image, histogram, matplotlib figure, and confusion-matrix logging.
python train_advanced.py

Write-Host "Done. Open https://wandb.ai to view the runs under project Week7-project."
