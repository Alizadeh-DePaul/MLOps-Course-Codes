#!/usr/bin/env nu
# Exercises/FastAPIApplication/demo.nu - cross-platform end-to-end runner.
# Run from inside Exercises/FastAPIApplication/ with a clean working tree.
#
# What this does (mirrors the exercise page):
#   1. Sets up a venv and installs the package + dev extras with uv
#   2. Runs the test suite (drives every endpoint in-process via TestClient)
#   3. Boots the API with uvicorn and hits two endpoints, then shuts it down
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

uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Run the test suite -------------------------------------------------
# No running server needed - FastAPI's TestClient calls the app directly.
pytest -v

# --- 3. Smoke-launch the API ----------------------------------------------
# Background uvicorn (job spawn needs Nushell 0.105+), give it a moment, hit
# two endpoints with the built-in `http get`, then stop the job.
# fastapi dev is the dev-time equivalent (with auto-reload).
print "Launching the API on http://127.0.0.1:8000 for a smoke check..."
let job_id = (job spawn { uvicorn app.main:app --host 127.0.0.1 --port 8000 })
sleep 4sec
http get http://127.0.0.1:8000/ | print
http get http://127.0.0.1:8000/items/1 | print
job kill $job_id
print "API smoke check complete."
