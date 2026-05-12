# Exercises/ApplicationLogging/demo.ps1 - Windows PowerShell end-to-end runner.
# Line-for-line mirror of demo.nu and demo.sh.
#
# Run from inside Exercises/ApplicationLogging/ with a clean working tree:
#   .\demo.ps1
#
# If Windows blocks the script, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What it does (short):
#   1. Creates a uv-managed venv and installs the package.
#   2. Runs logger_test.py (basic levels).
#   3. Runs logger_advanced.py (dictConfig + rotating file handlers).
#   4. Runs logging_rich.py (RichHandler + colorized console).
#   5. Runs logging_hydra.py (Hydra job_logging override).
#   6. Lists logs/ so you can see what got written.

$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once (Windows):
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Write-Host "[1/6] Creating venv + installing package ..."
uv venv                                       # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1                # Linux/Mac: source .venv/bin/activate
uv pip install -e .                           # alt: pip install -e .

# --- 2. Basic log levels ---------------------------------------------------
# Five log levels written straight to stdout with `basicConfig`. The
# cheapest possible logger - good for one-off scripts.
Write-Host ""
Write-Host "[2/6] logger_test.py (basic levels) ..."
python logger_test.py

# --- 3. dictConfig + rotating file handlers --------------------------------
# Adds two RotatingFileHandlers (info.log, error.log) layered on top of
# the console handler, all configured via `dictConfig`. Watch logs/ get
# populated after this step.
Write-Host ""
Write-Host "[3/6] logger_advanced.py (dictConfig + rotating files) ..."
python logger_advanced.py

# --- 4. RichHandler for colorized console ---------------------------------
# Same dictConfig, but the console handler is swapped for RichHandler
# so every level shows up colorized. The rich_tracebacks=True option
# also renders exceptions with the offending line highlighted.
Write-Host ""
Write-Host "[4/6] logging_rich.py (RichHandler + colorized output) ..."
python logging_rich.py

# --- 5. Hydra job_logging override ----------------------------------------
# Demonstrates that the same dictConfig schema can be supplied to Hydra
# via the `hydra.job_logging` key in conf/config.yaml. Each Hydra run
# gets its own outputs/<date>/<time>/main.log automatically.
Write-Host ""
Write-Host "[5/6] logging_hydra.py (Hydra job_logging override) ..."
python logging_hydra.py

# --- 6. Inspect generated logs --------------------------------------------
Write-Host ""
Write-Host "[6/6] Contents of logs/ after the demo:"
Get-ChildItem logs | Sort-Object Name | Format-Table -AutoSize
Write-Host ""
Write-Host "demo.ps1 finished. Open logs/info.log and logs/error.log to inspect."
