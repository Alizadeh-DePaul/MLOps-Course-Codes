#!/usr/bin/env nu
# Exercises/PerformanceProfiling/demo.nu - end-to-end runner for the Profiling exercise.
#
# Cross-platform: identical commands work on Windows, macOS, and Linux.
# Requires: nushell (https://www.nushell.sh) - install via:
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
#
# Run from inside Exercises/PerformanceProfiling/ with a clean working tree:
#   nu demo.nu
#
# What it does (short):
#   1. Creates a uv-managed venv with the cpu/cuda extra.
#   2. Runs cProfile on the VAE -> writes vae.prof.
#   3. Runs torch.profiler on a ResNet-18 forward pass -> log/resnet18/.
#   4. Runs torch.profiler on a full VAE training pass -> log/training/.
#   5. Prints viewer instructions (TensorBoard primary, Perfetto alternative).

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Defaults to the CPU torch wheel (~200 MB) so it works on every machine.
# Override by setting PROFILING_EXTRA=cuda before running, e.g.:
#   PROFILING_EXTRA=cuda nu demo.nu       # macOS / Linux
#   $env:PROFILING_EXTRA = "cuda"; nu demo.nu  # Windows PowerShell
print "[1/5] Syncing venv with the matching torch extra ..."
let extra = ($env | get -i PROFILING_EXTRA | default "cpu")
print $"  using --extra ($extra) and --extra viz"
uv sync $"--extra=($extra)" --extra viz

let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

# --- 2. cProfile -----------------------------------------------------------
# Forces epochs down to 1 in-memory so the smoke run finishes quickly.
print "\n[2/5] cProfile -> vae.prof (epochs forced to 1) ..."
(
    "import vae_mnist; vae_mnist.epochs = 1; "
    + "import cProfile, pstats; "
    + "p = cProfile.Profile(); p.enable(); vae_mnist.main(); p.disable(); "
    + "p.dump_stats('vae.prof'); "
    + "pstats.Stats('vae.prof').sort_stats('cumulative').print_stats(10)"
) | python -

# --- 3. ResNet18 forward pass ---------------------------------------------
print "\n[3/5] torch.profiler on ResNet-18 -> log/resnet18/ ..."
python profile_resnet.py

# --- 4. Full training profile ---------------------------------------------
print "\n[4/5] torch.profiler on VAE training -> log/training/ ..."
python profile_training.py

# --- 5. Viewer instructions ------------------------------------------------
print "\n[5/5] Done. Visualize the results with EITHER tool:"
print ""
print "  cProfile output:"
print "    snakeviz vae.prof"
print ""
print "  torch.profiler output (TensorBoard - primary):"
print "    tensorboard --logdir=./log"
print "    open http://localhost:6006/#pytorch_profiler"
print ""
print "  torch.profiler output (Perfetto - alternative for big traces or"
print "  when the TensorBoard plugin won't install):"
print "    open https://ui.perfetto.dev/ and drag in any .pt.trace.json"
print "    file from log/resnet18/ or log/training/"
