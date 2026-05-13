#!/usr/bin/env nu
# Exercises/WandB/demo.nu - cross-platform end-to-end runner for the WandB exercise.
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
$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (macOS/Linux)
#                  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  (Windows)
uv venv                                    # alt: python -m venv .venv

# Nushell doesn't source activation scripts - prepend the venv bin dir to PATH
# and set VIRTUAL_ENV ourselves. Works identically on Windows/macOS/Linux.
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

uv pip install -e .                        # alt: pip install -e .

# --- 2. Credential check ---------------------------------------------------
# Either ~/.netrc must be set up via `wandb login`, or WANDB_API_KEY must be in env.
# Bail out early with a clear message rather than letting wandb.init silently
# fall into anonymous mode or hang.
let netrc = if $nu.os-info.name == "windows" {
    ($env.USERPROFILE | path join ".netrc")
} else {
    ($env.HOME | path join ".netrc")
}
if not ((WANDB_API_KEY in $env) or ($netrc | path exists)) {
    print "ERROR: No W&B credentials found. Run `wandb login` first, or set WANDB_API_KEY."
    exit 1
}

# --- 3. Basic run ----------------------------------------------------------
# Logs `loss` as a scalar every 100 mini-batches.
python train.py

# --- 4. Advanced run -------------------------------------------------------
# Adds image, histogram, matplotlib figure, and confusion-matrix logging.
python train_advanced.py

print "Done. Open https://wandb.ai to view the runs under project Week7-project."
