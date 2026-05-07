#!/usr/bin/env bash
# Exercises/PerformanceProfiling/demo.sh - bash end-to-end runner.
#
# Run from inside Exercises/PerformanceProfiling/ with a clean working tree:
#   bash demo.sh
#
# Equivalent to demo.nu / demo.ps1.

set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Defaults to the CPU torch wheel (~200 MB) so it works on every machine.
# Override by exporting PROFILING_EXTRA=cuda before running:
#   PROFILING_EXTRA=cuda bash demo.sh
EXTRA="${PROFILING_EXTRA:-cpu}"
echo "[1/5] Syncing venv with --extra ${EXTRA} and --extra viz ..."
uv sync "--extra=${EXTRA}" --extra viz
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows Git Bash: .venv/Scripts/activate

# --- 2. cProfile -----------------------------------------------------------
echo
echo "[2/5] cProfile -> vae.prof (epochs forced to 1) ..."
python - <<'PY'
import vae_mnist
vae_mnist.epochs = 1
import cProfile, pstats
p = cProfile.Profile()
p.enable()
vae_mnist.main()
p.disable()
p.dump_stats("vae.prof")
pstats.Stats("vae.prof").sort_stats("cumulative").print_stats(10)
PY

# --- 3. ResNet18 forward pass ---------------------------------------------
echo
echo "[3/5] torch.profiler on ResNet-18 -> log/resnet18/ ..."
python profile_resnet.py

# --- 4. Full training profile ---------------------------------------------
echo
echo "[4/5] torch.profiler on VAE training -> log/training/ ..."
python profile_training.py

# --- 5. Viewer instructions -----------------------------------------------
cat <<'EOF'

[5/5] Done. Visualize the results with EITHER tool:

  cProfile output:
    snakeviz vae.prof

  torch.profiler output (TensorBoard - primary):
    tensorboard --logdir=./log
    open http://localhost:6006/#pytorch_profiler

  torch.profiler output (Perfetto - alternative for big traces or
  when the TensorBoard plugin won't install):
    open https://ui.perfetto.dev/ and drag in any .pt.trace.json
    file from log/resnet18/ or log/training/
EOF
