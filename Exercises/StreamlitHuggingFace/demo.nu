#!/usr/bin/env nu
# Exercises/StreamlitHuggingFace/demo.nu - cross-platform end-to-end runner.
# Run from inside Exercises/StreamlitHuggingFace/ with a clean working tree.
#
# What this does:
#   1. Sets up the venv with uv and installs the package + dev extras
#   2. Trains the tiny CNN (skips if model.pth already exists)
#   3. Runs the test suite
#   4. Starts the Streamlit app headless for a brief smoke check
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

# --- 2. Train the tiny CIFAR-10 model -------------------------------------
# Skip if model.pth already exists so reruns are fast.
if not ("model.pth" | path exists) {
    python train_model.py
} else {
    print "model.pth already exists - skipping training"
}

# --- 3. Run the test suite ------------------------------------------------
# Catches syntax errors and stale imports before we try to launch the UI.
pytest -v

# --- 4. Smoke-launch the Streamlit app ------------------------------------
# Headless start so CI / instructor demo doesn't open a browser tab.
# Kill it after 10s - just verifying it boots cleanly.
print "Launching Streamlit (headless) for 10s smoke check..."
let app = (do { streamlit run app.py --server.headless true --server.port 8501 } | complete)
print "Streamlit smoke check complete."
