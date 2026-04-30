# ML Code Debugging — Hunting Bugs in a PyTorch VAE

**Course:** SE 489 — MLOps (Week 5, ML Debugging and Performance Profiling)

A short, real-feeling debugging exercise. You're handed a 130-line VAE
training script (`vae_mnist_buggy.py`) that contains four bugs — one of
each common ML flavor: a device bug, a shape bug, a math bug, and a
training-loop bug. Your job is not just to fix them, but to *find* them
the way you would in a real codebase — with a debugger, not with print
statements.

You'll practice three debugging methods on the same script:

1. **`pdb` / `breakpoint()`** — the stdlib debugger; works everywhere.
2. **The VS Code Python Debugger** — `ms-python.debugpy`; the GUI debugger you'll actually use day-to-day.
3. **GitHub Copilot Chat** — `/fix`, `/explain`, and asking the AI questions about an exception. AI-assisted debugging is a real tool now; learn its strengths *and* its failure modes.

The exercise page has the full narrative, hint ladders, and tool walkthroughs.
This folder is the code you clone and edit.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `vae_mnist_buggy.py` | The buggy training script | **Yes** — fix the four bugs in place |
| `pyproject.toml` | Dependency pins for Python 3.11 | No |
| `.vscode/launch.json` | Pre-wired VS Code debugger config | No (but read it) |
| `.vscode/settings.json` | Minor VS Code settings for the project | No |
| `ai_debug_prompts.md` | Curated Copilot Chat prompts for ML debugging | Reference only |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end smoke test (run *after* you fix the script) | No |

## Quick start

This exercise targets **Python 3.11**. There are two install variants —
pick the one that matches your machine. They're mutually exclusive; pick
one or the other, not both.

| Your machine | Use this extra | Wheel size |
| --- | --- | --- |
| CPU-only laptop, Apple Silicon (Mac M-series) | `--extra cpu` | ~200 MB |
| Windows or Linux with NVIDIA GPU + drivers | `--extra cuda` | ~2.5 GB |

Don't know? Run `nvidia-smi`. If it prints a GPU table, use `cuda`. If
the command isn't found, use `cpu`.

```bash
# Install uv once (skip if you have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh             # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. Pick exactly ONE — sync your venv with the matching extra:
uv sync --extra cpu                  # CPU / Apple Silicon
# OR
uv sync --extra cuda                 # NVIDIA GPU (Windows / Linux)

# 2. Activate the venv:
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 3. Try to run the broken script (it will crash). That's the starting point.
python vae_mnist_buggy.py

# 4. Use a debugger to find each bug. See the exercise page for the workflow.
```

To add the dev tools (`ipdb`, `ruff`, `mypy`) on top:

```bash
uv sync --extra cuda --extra dev     # or: --extra cpu --extra dev
```

To switch later (e.g. you got a GPU): just re-run `uv sync` with the
other extra. uv will swap the wheels.

### Alternative (plain pip)

Plain pip doesn't read `[tool.uv.sources]`, so you have to point it at
the right index manually:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# CPU:
pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu

# CUDA:
pip install -e ".[cuda]" --extra-index-url https://download.pytorch.org/whl/cu128
```

## Three ways to debug — quick reference

```bash
# 1. pdb / breakpoint() — drop a line at the place you suspect
python -c "import sys; sys.breakpointhook"   # confirm pdb is reachable
# inside the script, add: breakpoint()
# run normally; the prompt drops you in. Useful commands:
#   n / s / c     step over / step into / continue
#   p expr        print value
#   pp expr       pretty-print
#   l             list source around the current line
#   w             where (full call stack)
#   q             quit
```

```bash
# 2. VS Code debugger — F5 from inside vae_mnist_buggy.py.
#    A pre-wired debugpy config is at .vscode/launch.json.
#    Set a breakpoint by clicking left of a line number (red dot).
```

```bash
# 3. Copilot Chat — open the panel (Ctrl+Alt+I / Cmd+Alt+I), select code,
#    type /fix or /explain. See ai_debug_prompts.md for prompts that
#    actually work for ML bugs.
```

### End-to-end dry run (after you've fixed the script)

Three equivalent runners are provided; pick whichever shell you prefer:

```nu
nu demo.nu           # cross-platform (Windows / macOS / Linux) - recommended
```

```bash
bash demo.sh         # macOS / Linux / WSL / Git Bash
```

```powershell
.\demo.ps1          # Windows PowerShell (no extra install needed)
```

Each runner installs the package, trains for 1 epoch (smoke test), and
verifies that `orig_data.png`, `reconstructions.png`, and
`generated_sample.png` are produced. They do NOT auto-fix the script —
you have to fix it first.

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the
> first time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

## Rules of the game

1. **Don't peek at the solution toggle on the exercise page until you've tried each tool.** The point is to practice the *process*, not to memorize four lines.
2. **Don't just take Copilot's first answer.** Copilot will sometimes propose fixes that paper over a bug rather than fix the root cause. Verify with the debugger before accepting.
3. **Fix one bug at a time.** Re-run the script after each fix. The bugs hide each other — fixing the shape bug exposes the math bug, fixing the math bug exposes the training bug, and so on.

## Gotchas

- **PyTorch on Apple Silicon**: if you're on an M-series Mac, set `cuda = False` in the script (or use the device-detection idiom you'll discover while fixing the device bug). MPS works for this exercise, but most CUDA tutorials assume an NVIDIA GPU.
- **MNIST download**: the script downloads MNIST to `./datasets/` on first run. About 60 MB; takes ~30 s on a normal connection.
- **`epochs=20` is too long for class.** While you're iterating, drop `epochs` down to 1 or 2 so each run finishes in seconds. Bump it back up only when you think the script is correct.
- **Loss not going down?** That's a *real* bug, not a tooling problem. Don't dismiss it as "neural networks are noisy."
