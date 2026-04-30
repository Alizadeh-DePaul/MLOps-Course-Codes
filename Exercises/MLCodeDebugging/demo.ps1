# Exercises/MLCodeDebugging/demo.ps1 - Windows PowerShell end-to-end runner.
#
# Run AFTER you've fixed the four bugs in vae_mnist_buggy.py:
#   .\demo.ps1
#
# If Windows blocks execution the first time, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# Equivalent to demo.nu / demo.sh.

$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Write-Host "[1/4] Creating venv + installing package ..."
uv venv                                                   # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e .                                       # alt: pip install -e .

# --- 2. Run the (fixed) script with epochs forced to 1 ---------------------
Write-Host ""
Write-Host "[2/4] Running fixed script (epochs forced to 1 for smoke test) ..."
$pyCode = @'
src = open("vae_mnist_buggy.py").read().replace("epochs = 20", "epochs = 1")
exec(compile(src, "vae_mnist_buggy.py", "exec"), {"__name__": "__main__"})
'@
$pyCode | python -

# --- 3. Verify outputs -----------------------------------------------------
Write-Host ""
Write-Host "[3/4] Checking that the three expected PNGs were produced ..."
foreach ($f in @("orig_data.png", "reconstructions.png", "generated_sample.png")) {
    if (-not (Test-Path $f)) {
        Write-Host "  FAIL: $f was not produced"
        exit 1
    }
    Write-Host "  OK: $f"
}

# --- 4. Done ---------------------------------------------------------------
Write-Host ""
Write-Host "[4/4] demo.ps1 finished. Open the three PNGs to inspect the model output."
