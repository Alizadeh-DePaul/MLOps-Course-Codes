#!/usr/bin/env nu
# Exercises/GitHubActions/demo.nu - cross-platform end-to-end runner.
#
# Identical commands work on Windows, macOS, and Linux. Requires nushell:
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
#
# What it does:
#   1. Syncs a uv-managed venv from uv.lock.
#   2. Runs pytest locally so you see the same green/red CI will see.
#   3. Runs ruff and mypy locally (mirrors the codecheck workflow).
#   4. (Optional) Runs the GitHub workflows locally with `act` if available,
#      so you can catch YAML issues without burning CI minutes.
#
# Run it from inside this folder:
#   nu demo.nu

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
print "[1/4] Syncing venv from uv.lock ..."
uv sync --frozen --extra dev

# Activate-by-PATH (nushell doesn't source activation scripts).
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

# --- 2. Run the tests ------------------------------------------------------
print "\n[2/4] Running pytest ..."
uv run pytest -v

# --- 3. Code checks --------------------------------------------------------
print "\n[3/4] Running ruff + mypy ..."
uv run ruff check .
uv run ruff format --check .
uv run mypy simple_mlops

# --- 4. (Optional) Run workflows locally with act --------------------------
# `act` is a tool that runs GitHub Actions workflows on your machine using
# Docker. If you have Docker Desktop running and `act` installed, you can
# catch syntax errors without burning CI minutes.
#   Install: winget install nektos.act / brew install act / cargo install act
print "\n[4/4] (Optional) Local workflow validation with act ..."
let has_act = (which act | length) > 0
if $has_act {
    print "  act found - running .github/workflows/tests.yaml locally"
    act push --workflows .github/workflows/tests.yaml --container-architecture linux/amd64
} else {
    print "  act not installed - skipping local workflow run."
    print "  Install with: winget install nektos.act (Windows) | brew install act (macOS) | cargo install act"
}
