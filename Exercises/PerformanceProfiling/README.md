# Python and ML Code Performance Profiling

**Course:** SE 489 — MLOps (Week 5/6, ML Debugging and Performance Profiling)

A short hands-on exercise on profiling. You start from a clean (already-debugged)
PyTorch VAE training script and use four tools to figure out where time is
actually being spent: `cProfile`, `snakeviz`, `torch.profiler`, and either
TensorBoard *or* Perfetto UI for visualization.

The exercise has two halves:

1. **`cProfile`** — Python's stdlib profiler. Function-level timing for any
   Python script. Visualize with `snakeviz`.
2. **`torch.profiler`** — PyTorch's built-in profiler. Captures CPU+GPU work,
   memory, kernel times, and operator dispatch. Visualize with TensorBoard
   (primary) *or* Perfetto UI (alternative for large traces or when the
   TensorBoard plugin won't install).

The exercise page has the full narrative and questions to answer at each step.
This folder is the code you clone and edit.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `vae_mnist.py` | Clean VAE training script (the previous exercise's solution) | Maybe — you'll optimize it in the last cProfile step |
| `profile_cprofile.py` | Section 1 driver — runs `vae_mnist.py` under cProfile | **Yes** — work through the TODOs |
| `profile_resnet.py` | Section 2 driver — `torch.profiler` on a ResNet-18 forward pass | **Yes** — work through the TODOs |
| `profile_training.py` | Section 2, final step — `torch.profiler` on a full training run | **Yes** — answer the bottleneck question |
| `pyproject.toml` | Dependency pins for Python 3.11 (cpu/cuda/viz/dev extras) | No |
| `.vscode/launch.json` | Pre-wired VS Code debugger configs for each script | No (but read it) |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner that produces all profile artifacts | No |

## Quick start

This exercise targets **Python 3.11**. There are two install variants for
PyTorch — pick the one that matches your machine. They're mutually exclusive.

| Your machine | Use this extra | Wheel size |
| --- | --- | --- |
| CPU-only laptop, Apple Silicon (Mac M-series) | `--extra cpu` | ~200 MB |
| Windows or Linux with NVIDIA GPU + drivers | `--extra cuda` | ~2.5 GB |

If you don't know, run `nvidia-smi`. GPU table → `cuda`. Command not found → `cpu`.

You also want the `viz` extra for the TensorBoard viewer:

```bash
# Install uv once (skip if you have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh             # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. Pick exactly ONE of cpu/cuda; add viz for the TensorBoard plugin:
uv sync --extra cpu --extra viz             # CPU / Apple Silicon
# OR
uv sync --extra cuda --extra viz            # NVIDIA GPU (Windows / Linux)

# 2. Activate the venv:
source .venv/bin/activate                   # Windows: .venv\Scripts\Activate.ps1

# 3. Run each section's driver in order:
python profile_cprofile.py        # writes vae.prof + prints top-10 table
snakeviz vae.prof                 # interactive cProfile viewer in browser

python profile_resnet.py          # writes log/resnet18/ + trace.json
python profile_training.py        # writes log/training/

# 4. Visualize the torch.profiler output (pick ONE):
tensorboard --logdir=./log        # then open http://localhost:6006/#pytorch_profiler
# OR open https://ui.perfetto.dev/ and drag in any log/**/*.pt.trace.json
```

### Alternative (plain pip)

`pip` doesn't read `[tool.uv.sources]`, so the index has to be passed explicitly:

```bash
python -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\Activate.ps1

# CPU:
pip install -e ".[cpu,viz]" --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA:
pip install -e ".[cuda,viz]" --extra-index-url https://download.pytorch.org/whl/cu128
```

## Visualizing the profiler output

You have two choices for `torch.profiler` output. Both consume the same
`.pt.trace.json` file. Pick whichever works for you.

**TensorBoard (primary).** What the exercise page walks through. Comes from
the `torch-tb-profiler` plugin (the `viz` extra above). Run:

```bash
tensorboard --logdir=./log
```

Then open `http://localhost:6006/#pytorch_profiler`.

> Heads-up: the TensorBoard plugin (`torch-tb-profiler`) is in maintenance
> mode and PyTorch upstream has marked TensorBoard plugin support deprecated.
> The plugin still works as of 2026, but install can occasionally fail with
> certain `uv` builds (astral-sh/uv#16651). If `uv sync --extra viz` fails,
> try `uv pip install torch-tb-profiler --no-build-isolation`, or skip
> straight to Perfetto below.

**Perfetto UI (alternative).** No install — it's a web app. Better for very
large traces. Open <https://ui.perfetto.dev/> and drag any
`log/**/*.pt.trace.json` file onto the page. The same trace files work in
both viewers.

## Three ways to run the demo

```nu
nu demo.nu           # cross-platform (Windows / macOS / Linux) - recommended
```

```bash
bash demo.sh         # macOS / Linux / WSL / Git Bash
```

```powershell
.\demo.ps1           # Windows PowerShell (no extra install needed)
```

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.
>
> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

## Going further

After the in-class exercise, two related tools are worth a look:

- **`line_profiler` / `kernprof`** — line-by-line timing inside a known hot
  function. Great for the "I know which function is slow but not which
  *line*" case. <https://github.com/pyutils/line_profiler>
- **`py-spy`** — sampling profiler that attaches to a running Python
  process. No code change, no restart, low overhead. Good for diagnosing
  long-running training jobs in production. <https://github.com/benfred/py-spy>
- **`memray`** — Python memory profiler from Bloomberg, complementary to
  the CPU-side tools above. <https://github.com/bloomberg/memray>
- **Holistic Trace Analysis (HTA)** — for distributed training workloads,
  consumes the same `.pt.trace.json` files. Heavyweight; Linux/Mac only.
  <https://github.com/facebookresearch/HolisticTraceAnalysis>
