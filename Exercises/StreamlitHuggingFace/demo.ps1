# Exercises/StreamlitHuggingFace/demo.ps1 - Windows PowerShell end-to-end runner.
# Run from inside Exercises/StreamlitHuggingFace/ with a clean working tree.
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv                                    # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Train the tiny CIFAR-10 model -------------------------------------
# Skip if model.pth already exists so reruns are fast.
if (-not (Test-Path "model.pth")) {
    python train_model.py
} else {
    Write-Host "model.pth already exists - skipping training"
}

# --- 3. Run the test suite ------------------------------------------------
pytest -v

# --- 4. Smoke-launch the Streamlit app ------------------------------------
# Headless start, kill after 10s. Just verifying the app boots cleanly.
Write-Host "Launching Streamlit (headless) for 10s smoke check..."
$app = Start-Process -FilePath "streamlit" `
    -ArgumentList "run", "app.py", "--server.headless", "true", "--server.port", "8501" `
    -NoNewWindow -PassThru
Start-Sleep -Seconds 10
Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
Write-Host "Streamlit smoke check complete."
