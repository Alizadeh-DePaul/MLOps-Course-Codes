#!/usr/bin/env nu
# Exercises/MLCodeDebugging/demo.nu - end-to-end runner for the ML Code Debugging exercise.
#
# Cross-platform: identical commands work on Windows, macOS, and Linux.
# Requires: nushell (https://www.nushell.sh) - install via:
#   winget install nushell           # Windows
#   brew install nushell             # macOS
#   cargo install nu                 # any platform
#
# Run AFTER you've fixed the four bugs in vae_mnist_buggy.py:
#   nu demo.nu
#
# What it does (short):
#   1. Creates a uv-managed venv and installs the package.
#   2. Trains for 1 epoch on MNIST as a smoke test.
#   3. Verifies the three output PNGs were written.
#
# This runner does NOT auto-fix the script. If your fixes are missing or
# wrong, the script will crash here and the runner will exit non-zero.

$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once:
#   Windows:     powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
#   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# This runner defaults to the CPU torch wheel (~200 MB) so it works on every
# machine. Override by setting MLCODEDEBUG_EXTRA=cuda before running, e.g.:
#   MLCODEDEBUG_EXTRA=cuda nu demo.nu       # macOS / Linux
#   $env:MLCODEDEBUG_EXTRA = "cuda"; nu demo.nu  # Windows PowerShell
print "[1/4] Syncing venv with the matching torch extra ..."
let extra = ($env | get -i MLCODEDEBUG_EXTRA | default "cpu")
print $"  using --extra ($extra)"
uv sync $"--extra=($extra)"

let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

# --- 2. Patch epochs down so the smoke test finishes quickly ---------------
# The script ships with epochs=20. For a smoke test we want one epoch.
# We do this in-memory via a temp env var inside a one-liner Python script.
print "\n[2/4] Running fixed script (epochs forced to 1 for smoke test) ..."
(
    "import runpy, sys; "
    + "src = open('vae_mnist_buggy.py').read().replace('epochs = 20', 'epochs = 1'); "
    + "exec(compile(src, 'vae_mnist_buggy.py', 'exec'), {'__name__': '__main__'})"
) | python -

# --- 3. Verify outputs -----------------------------------------------------
print "\n[3/4] Checking that the three expected PNGs were produced ..."
let outputs = ["orig_data.png", "reconstructions.png", "generated_sample.png"]
for f in $outputs {
    if not ($f | path exists) {
        print $"  FAIL: ($f) was not produced"
        exit 1
    }
    print $"  OK: ($f)"
}

# --- 4. Done ---------------------------------------------------------------
print "\n[4/4] demo.nu finished. Open the three PNGs to inspect the model output."
