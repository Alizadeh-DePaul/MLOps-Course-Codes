#!/usr/bin/env bash
# Exercises/MLCodeDebugging/demo.sh - bash end-to-end runner.
#
# Run AFTER you've fixed the four bugs in vae_mnist_buggy.py:
#   bash demo.sh
#
# Equivalent to demo.nu / demo.ps1.

set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Defaults to the CPU torch wheel (~200 MB) so it works on every machine.
# Override by exporting MLCODEDEBUG_EXTRA=cuda before running:
#   MLCODEDEBUG_EXTRA=cuda bash demo.sh
EXTRA="${MLCODEDEBUG_EXTRA:-cpu}"
echo "[1/4] Syncing venv with --extra ${EXTRA} ..."
uv sync "--extra=${EXTRA}"
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows Git Bash: .venv/Scripts/activate

# --- 2. Run the (fixed) script with epochs forced to 1 ---------------------
echo
echo "[2/4] Running fixed script (epochs forced to 1 for smoke test) ..."
python - <<'PY'
src = open("vae_mnist_buggy.py").read().replace("epochs = 20", "epochs = 1")
exec(compile(src, "vae_mnist_buggy.py", "exec"), {"__name__": "__main__"})
PY

# --- 3. Verify outputs -----------------------------------------------------
echo
echo "[3/4] Checking that the three expected PNGs were produced ..."
for f in orig_data.png reconstructions.png generated_sample.png; do
    if [[ ! -f "$f" ]]; then
        echo "  FAIL: $f was not produced"
        exit 1
    fi
    echo "  OK: $f"
done

# --- 4. Done ---------------------------------------------------------------
echo
echo "[4/4] demo.sh finished. Open the three PNGs to inspect the model output."
