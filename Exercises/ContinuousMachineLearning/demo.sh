#!/usr/bin/env bash
# Exercises/ContinuousMachineLearning/demo.sh - bash end-to-end runner.
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
# Run it from inside this folder:
#   bash demo.sh
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
echo "[1/3] Creating venv and installing deps with uv ..."
uv venv                                    # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows Git Bash: source .venv/Scripts/activate
uv pip install -e .                        # alt: pip install -e .

# --- 2. Train the model ----------------------------------------------------
echo ""
echo "[2/3] Training model ..."
python simple_mlops/train_model_cml.py

echo ""
echo "      Artifacts produced:"
ls -la classification_report.txt confusion_matrix.png

# --- 3. (Optional) Validate the workflow locally with act ------------------
# `act` runs GitHub Actions workflows on your machine using Docker. Useful
# for catching YAML syntax errors without burning CI minutes.
echo ""
echo "[3/3] (Optional) Local workflow validation with act ..."
if command -v act >/dev/null 2>&1; then
    echo "  act found - running .github/workflows/cml.yaml locally"
    act push --workflows .github/workflows/cml.yaml --container-architecture linux/amd64
else
    echo "  act not installed - skipping local workflow run."
    echo "  Install with: brew install act (macOS) | cargo install act | winget install nektos.act (Windows)"
fi
