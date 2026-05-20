#!/usr/bin/env nu
# Exercises/PreCommit/demo.nu - cross-platform end-to-end runner for the
# Pre-commit exercise. Run from inside Exercises/PreCommit/ with a clean
# working tree.
#
# What this does (mirrors the exercise page steps 1-10):
#   1. Creates a venv and installs pre-commit (with the pre-commit-uv plugin)
#   2. Installs the hooks into .git/hooks
#   3. Runs every hook against every file (the four default sample hooks)
#   4. Demonstrates `pre-commit autoupdate` to bump rev tags
#   5. Demonstrates `pre-commit uninstall` to remove the hooks
$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh   (macOS/Linux)
#                  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  (Windows)
uv venv                                            # alt: python -m venv .venv

# Nushell doesn't source activation scripts - prepend the venv bin dir to PATH
# and set VIRTUAL_ENV ourselves. Works identically on Windows/macOS/Linux.
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

# --- 2. Install pre-commit -------------------------------------------------
# The pre-commit-uv plugin makes Python-based hooks ~30% faster to install.
uv tool install pre-commit --with pre-commit-uv    # alt: pip install pre-commit

# --- 3. Install hooks into .git/hooks --------------------------------------
# This must be run inside a git repo. We use the top-level course repo.
pre-commit install

# --- 4. Run every hook against every file ----------------------------------
# Two of these will fail and auto-fix sample_code.py:
#   - trailing-whitespace strips the spaces on lines 7 and 13
#   - end-of-file-fixer adds the missing final newline
# Re-running passes once the fixes are staged.
pre-commit run --all-files

# --- 5. Autoupdate ---------------------------------------------------------
# Bumps every `rev:` in .pre-commit-config.yaml to the latest tag on the
# default branch of each hook repo. In a real project, commit the resulting
# diff separately from feature work.
pre-commit autoupdate

# --- 6. Uninstall ----------------------------------------------------------
# How to disable pre-commit if you grow weary of it.
pre-commit uninstall
