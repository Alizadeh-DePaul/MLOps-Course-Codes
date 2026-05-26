#!/usr/bin/env bash
# Exercises/StreamlitHuggingFace/demo.sh - bash end-to-end runner.
# Run from inside Exercises/StreamlitHuggingFace/ with a clean working tree.
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (or PowerShell variant on Windows)
uv venv                                    # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Train the tiny CIFAR-10 model -------------------------------------
# Skip if model.pth already exists so reruns are fast.
if [ ! -f model.pth ]; then
    python train_model.py
else
    echo "model.pth already exists - skipping training"
fi

# --- 3. Run the test suite ------------------------------------------------
pytest -v

# --- 4. Smoke-launch the Streamlit app ------------------------------------
# Headless start, kill after 10s. Just verifying the app boots cleanly.
echo "Launching Streamlit (headless) for 10s smoke check..."
streamlit run app.py --server.headless true --server.port 8501 &
APP_PID=$!
sleep 10
kill "$APP_PID" 2>/dev/null || true
echo "Streamlit smoke check complete."
