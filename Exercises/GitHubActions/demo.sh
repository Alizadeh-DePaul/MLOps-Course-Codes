#!/usr/bin/env bash
# Exercises/GitHubActions/demo.sh - bash end-to-end runner.
#
# What it does:
#   1. Syncs a uv-managed venv from uv.lock.
#   2. Runs pytest locally so you see the same green/red CI will see.
#   3. Runs ruff and mypy locally (mirrors the codecheck workflow).
#   4. (Optional) Runs the GitHub workflows locally with `act` if available.
#
# Run it from inside this folder:
#   bash demo.sh
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
echo "[1/4] Syncing venv from uv.lock ..."
uv sync --frozen --extra dev
# shellcheck disable=SC1091
source .venv/bin/activate

# --- 2. Run the tests ------------------------------------------------------
echo
echo "[2/4] Running pytest ..."
uv run pytest -v

# --- 3. Code checks --------------------------------------------------------
echo
echo "[3/4] Running ruff + mypy ..."
uv run ruff check .
uv run ruff format --check .
uv run mypy simple_mlops

# --- 4. (Optional) Run workflows locally with act --------------------------
# `act` runs GitHub Actions workflows on your machine using Docker. Catch
# YAML/version issues here instead of burning CI minutes.
#   Install: brew install act / cargo install act
echo
echo "[4/4] (Optional) Local workflow validation with act ..."
if command -v act >/dev/null 2>&1; then
    echo "  act found - running .github/workflows/tests.yaml locally"
    act push --workflows .github/workflows/tests.yaml --container-architecture linux/amd64
else
    echo "  act not installed - skipping local workflow run."
    echo "  Install with: brew install act (macOS) | cargo install act (any) | https://nektosact.com/"
fi
