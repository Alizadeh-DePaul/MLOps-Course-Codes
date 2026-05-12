#!/usr/bin/env nu
# Exercises/ApplicationLogging/demo.nu - cross-platform end-to-end runner.
#
# Same phases as demo.sh and demo.ps1, just in nushell so a single file
# runs identically on Windows, macOS, and Linux.
#
# Install nushell once (any platform):
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
#
# Run from inside Exercises/ApplicationLogging/ with a clean working tree:
#   nu demo.nu
#
# What it does (short):
#   1. Creates a uv-managed venv and installs the package.
#   2. Runs logger_test.py (basic levels).
#   3. Runs logger_advanced.py (dictConfig + rotating file handlers).
#   4. Runs logging_rich.py (RichHandler + colorized console).
#   5. Runs logging_hydra.py (Hydra job_logging override).
#   6. Lists logs/ so you can see what got written.

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
print "[1/6] Creating venv + installing package ..."
uv venv                                    # alt: python -m venv .venv

# Activate the venv by prepending its bin dir to PATH (nu-native; no
# activate-script sourcing needed, works identically on every OS).
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

uv pip install -e .                        # alt: pip install -e .

# --- 2. Basic log levels ---------------------------------------------------
# Five log levels written straight to stdout with `basicConfig`. The
# cheapest possible logger - good for one-off scripts.
print "\n[2/6] logger_test.py (basic levels) ..."
python logger_test.py

# --- 3. dictConfig + rotating file handlers --------------------------------
# Adds two RotatingFileHandlers (info.log, error.log) layered on top of
# the console handler, all configured via `dictConfig`. Watch logs/ get
# populated after this step.
print "\n[3/6] logger_advanced.py (dictConfig + rotating files) ..."
python logger_advanced.py

# --- 4. RichHandler for colorized console ---------------------------------
# Same dictConfig, but the console handler is swapped for RichHandler
# so every level shows up colorized. The rich_tracebacks=True option
# also renders exceptions with the offending line highlighted.
print "\n[4/6] logging_rich.py (RichHandler + colorized output) ..."
python logging_rich.py

# --- 5. Hydra job_logging override ----------------------------------------
# Demonstrates that the same dictConfig schema can be supplied to Hydra
# via the `hydra.job_logging` key in conf/config.yaml. Each Hydra run
# gets its own outputs/<date>/<time>/main.log automatically.
print "\n[5/6] logging_hydra.py (Hydra job_logging override) ..."
python logging_hydra.py

# --- 6. Inspect generated logs --------------------------------------------
print "\n[6/6] Contents of logs/ after the demo:"
ls logs | sort-by name
print "\ndemo.nu finished. Open logs/info.log and logs/error.log to inspect."
