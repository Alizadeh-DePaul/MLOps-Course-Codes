#!/usr/bin/env nu
# Exercises/ContinuousMachineLearning/demo.nu - cross-platform end-to-end runner.
#
# Identical commands work on Windows, macOS, and Linux. Requires nushell:
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
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
#   nu demo.nu

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
print "[1/3] Creating venv and installing deps with uv ..."
uv venv

# Activate-by-PATH (nushell doesn't source activation scripts).
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

uv pip install -e .                        # alt: pip install -e .

# --- 2. Train the model ----------------------------------------------------
# Produces:
#   - classification_report.txt
#   - confusion_matrix.png
# These are the two artifacts the GitHub Actions workflow will embed in
# the PR comment when CI runs the same training step.
print "\n[2/3] Training model ..."
python simple_mlops/train_model_cml.py

print "\n      Artifacts produced:"
ls classification_report.txt confusion_matrix.png | select name size

# --- 3. (Optional) Validate the workflow locally with act ------------------
# `act` runs GitHub Actions workflows on your machine using Docker. Useful
# for catching YAML syntax errors without burning CI minutes.
#   Install: winget install nektos.act / brew install act / cargo install act
print "\n[3/3] (Optional) Local workflow validation with act ..."
let has_act = (which act | length) > 0
if $has_act {
    print "  act found - running .github/workflows/cml.yaml locally"
    act push --workflows .github/workflows/cml.yaml --container-architecture linux/amd64
} else {
    print "  act not installed - skipping local workflow run."
    print "  Install with: winget install nektos.act (Windows) | brew install act (macOS) | cargo install act"
}
