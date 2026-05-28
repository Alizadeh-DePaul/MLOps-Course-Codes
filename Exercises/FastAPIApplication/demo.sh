#!/usr/bin/env bash
# Exercises/FastAPIApplication/demo.sh - bash end-to-end runner.
# Run from inside Exercises/FastAPIApplication/ with a clean working tree.
#
# What this does (mirrors the exercise page):
#   1. Sets up a venv and installs the package + dev extras with uv
#   2. Runs the test suite (drives every endpoint in-process via TestClient)
#   3. Boots the API with uvicorn and curls two endpoints, then shuts it down
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
#                  (PowerShell variant on Windows)
uv venv                                    # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Run the test suite -------------------------------------------------
# No running server needed - FastAPI's TestClient calls the app directly.
pytest -v

# --- 3. Smoke-launch the API ----------------------------------------------
# Background uvicorn on port 8000, give it a moment, curl two endpoints,
# then stop it. fastapi dev is the dev-time equivalent (with auto-reload).
echo "Launching the API on http://127.0.0.1:8000 for a smoke check..."
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
sleep 4
curl -s http://127.0.0.1:8000/ && echo
curl -s http://127.0.0.1:8000/items/1 && echo
kill "$API_PID" 2>/dev/null || true
echo "API smoke check complete."
