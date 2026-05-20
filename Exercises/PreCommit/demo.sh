#!/usr/bin/env bash
# Exercises/PreCommit/demo.sh - bash end-to-end runner for the Pre-commit
# exercise. Run from inside Exercises/PreCommit/ with a clean working tree.
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
# (or PowerShell variant on Windows)
uv venv                                            # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                          # Windows: .venv\Scripts\activate

# --- 2. Install pre-commit -------------------------------------------------
# pre-commit-uv plugin makes Python-based hooks ~30% faster to install.
uv tool install pre-commit --with pre-commit-uv    # alt: pip install pre-commit

# --- 3. Install hooks into .git/hooks --------------------------------------
pre-commit install

# --- 4. Run every hook against every file ----------------------------------
# trailing-whitespace and end-of-file-fixer will fix sample_code.py on the
# first pass. A second `--all-files` run should be clean.
pre-commit run --all-files

# --- 5. Autoupdate ---------------------------------------------------------
# Bumps every `rev:` in .pre-commit-config.yaml to the latest tag on the
# default branch of each hook repo.
pre-commit autoupdate

# --- 6. Uninstall ----------------------------------------------------------
pre-commit uninstall
