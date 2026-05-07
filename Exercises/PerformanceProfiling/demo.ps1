# Exercises/PerformanceProfiling/demo.ps1 - Windows PowerShell end-to-end runner.
#
# Run from inside Exercises/PerformanceProfiling/ with a clean working tree:
#   .\demo.ps1
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# Equivalent to demo.nu / demo.sh.

$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#
# Defaults to the CPU torch wheel (~200 MB). Override by setting
# $env:PROFILING_EXTRA = "cuda" before running.
$extra = if ($env:PROFILING_EXTRA) { $env:PROFILING_EXTRA } else { "cpu" }
Write-Host "[1/5] Syncing venv with --extra $extra and --extra viz ..."
uv sync "--extra=$extra" --extra viz

. .\.venv\Scripts\Activate.ps1

# --- 2. cProfile -----------------------------------------------------------
Write-Host ""
Write-Host "[2/5] cProfile -> vae.prof (epochs forced to 1) ..."
$cprofileScript = @'
import vae_mnist
vae_mnist.epochs = 1
import cProfile, pstats
p = cProfile.Profile()
p.enable()
vae_mnist.main()
p.disable()
p.dump_stats("vae.prof")
pstats.Stats("vae.prof").sort_stats("cumulative").print_stats(10)
'@
$cprofileScript | python -

# --- 3. ResNet18 forward pass ---------------------------------------------
Write-Host ""
Write-Host "[3/5] torch.profiler on ResNet-18 -> log/resnet18/ ..."
python profile_resnet.py

# --- 4. Full training profile ---------------------------------------------
Write-Host ""
Write-Host "[4/5] torch.profiler on VAE training -> log/training/ ..."
python profile_training.py

# --- 5. Viewer instructions -----------------------------------------------
Write-Host ""
Write-Host "[5/5] Done. Visualize the results with EITHER tool:"
Write-Host ""
Write-Host "  cProfile output:"
Write-Host "    snakeviz vae.prof"
Write-Host ""
Write-Host "  torch.profiler output (TensorBoard - primary):"
Write-Host "    tensorboard --logdir=./log"
Write-Host "    open http://localhost:6006/#pytorch_profiler"
Write-Host ""
Write-Host "  torch.profiler output (Perfetto - alternative for big traces or"
Write-Host "  when the TensorBoard plugin won't install):"
Write-Host "    open https://ui.perfetto.dev/ and drag in any .pt.trace.json"
Write-Host "    file from log/resnet18/ or log/training/"
