# Exercises/PreCommit/demo.ps1 - Windows PowerShell end-to-end runner for the
# Pre-commit exercise. Run from inside Exercises/PreCommit/ with a clean
# working tree.
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv                                            # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1

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
