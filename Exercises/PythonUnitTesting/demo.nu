#!/usr/bin/env nu
# Exercises/PythonUnitTesting/demo.nu - cross-platform end-to-end runner.
#
# Identical commands work on Windows, macOS, and Linux. Requires nushell:
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
#
# Run AFTER you've filled in the TODOs in tests/*.py:
#   nu demo.nu
#
# What it does:
#   1. Syncs a uv-managed venv with the matching torch extra.
#   2. Runs pytest (with verbose output).
#   3. Re-runs pytest under pytest-cov and prints the coverage summary.
#   4. Writes an HTML coverage report to htmlcov/ and prints its path.

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Defaults to the CPU torch wheel (~200 MB). Override with:
#   PYUT_EXTRA=cuda nu demo.nu                  # macOS / Linux
#   $env:PYUT_EXTRA = "cuda"; nu demo.nu        # Windows PowerShell
print "[1/4] Syncing venv with the matching torch extra ..."
let extra = ($env | get -i PYUT_EXTRA | default "cpu")
print $"  using --extra ($extra)"
uv sync $"--extra=($extra)"

# Activate-by-PATH (nushell doesn't source activation scripts).
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

# --- 2. Run the tests ------------------------------------------------------
# Whatever you've filled in for the TODOs will run here. Any unsolved
# test will raise NotImplementedError and fail the run.
print "\n[2/4] Running pytest ..."
pytest -v

# --- 3. Run with coverage --------------------------------------------------
# Sub-exercise 7 — code coverage via pytest-cov.
print "\n[3/4] Re-running under pytest-cov ..."
pytest --cov=models --cov=training --cov-report=term-missing --cov-report=html

# --- 4. Point at the HTML report -------------------------------------------
print "\n[4/4] HTML coverage report:"
let report = (pwd | path join "htmlcov" "index.html")
print $"  ($report)"
print "  Open it in a browser to see line-by-line coverage."
