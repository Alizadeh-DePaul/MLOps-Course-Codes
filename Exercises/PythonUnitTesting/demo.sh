#!/usr/bin/env bash
# Exercises/PythonUnitTesting/demo.sh - bash end-to-end runner.
#
# Run AFTER you've filled in the TODOs in tests/*.py:
#   bash demo.sh
#
# Equivalent to demo.nu / demo.ps1.

set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Defaults to the CPU torch wheel (~200 MB). Override with:
#   PYUT_EXTRA=cuda bash demo.sh
EXTRA="${PYUT_EXTRA:-cpu}"
echo "[1/4] Syncing venv with --extra ${EXTRA} ..."
uv sync "--extra=${EXTRA}"
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows Git Bash: .venv/Scripts/activate

# --- 2. Run the tests ------------------------------------------------------
echo
echo "[2/4] Running pytest ..."
pytest -v

# --- 3. Run with coverage --------------------------------------------------
echo
echo "[3/4] Re-running under pytest-cov ..."
pytest --cov=models --cov=training --cov-report=term-missing --cov-report=html

# --- 4. Point at the HTML report -------------------------------------------
echo
echo "[4/4] HTML coverage report:"
echo "  $(pwd)/htmlcov/index.html"
echo "  Open it in a browser to see line-by-line coverage."
