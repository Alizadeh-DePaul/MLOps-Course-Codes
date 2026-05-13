#!/usr/bin/env bash
# Exercises/WandB/demo.sh - bash end-to-end runner for the WandB exercise.
# Run from inside Exercises/WandB/ after `wandb login` (or with WANDB_API_KEY set).
#
# What this does (mirrors the exercise page steps 2-6):
#   1. Creates a venv with uv and installs the package
#   2. Confirms WandB credentials are available
#   3. Runs train.py (basic scalar logging)
#   4. Runs train_advanced.py (images / histograms / matplotlib / confusion matrix)
#
# This script does NOT run `wandb agent` because sweep agents block forever.
# See the README for the manual sweep flow.
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (or PowerShell variant on Windows)
uv venv                                    # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
uv pip install -e .                        # alt: pip install -e .

# --- 2. Credential check ---------------------------------------------------
# Bail out early if neither WANDB_API_KEY nor ~/.netrc is set up.
if [[ -z "${WANDB_API_KEY:-}" && ! -f "${HOME}/.netrc" ]]; then
    echo "ERROR: No W&B credentials found. Run \`wandb login\` first, or set WANDB_API_KEY." >&2
    exit 1
fi

# --- 3. Basic run ----------------------------------------------------------
# Logs `loss` as a scalar every 100 mini-batches.
python train.py

# --- 4. Advanced run -------------------------------------------------------
# Adds image, histogram, matplotlib figure, and confusion-matrix logging.
python train_advanced.py

echo "Done. Open https://wandb.ai to view the runs under project Week7-project."
