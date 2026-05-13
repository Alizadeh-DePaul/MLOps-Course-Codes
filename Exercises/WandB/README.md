# WandB — Experiment Logging in Python

**Course:** SE 489 — MLOps (Week 7, Experiment Tracking)

This package is the starter scaffold for the Weights & Biases experiment-logging
exercise. You'll log scalars from a small MNIST training loop, then upgrade to
non-scalar logging (images, histograms, matplotlib figures, confusion matrices),
run a hyperparameter sweep, and finally containerise the same training script
and authenticate W&B via an environment variable.

Follow the exercise page for the step-by-step narrative. The files here are
what you actually edit and run.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `train.py` | Minimal MNIST loop logging `loss` to W&B via `wandb.init` / `wandb.log` / `wandb.finish` | **Yes** — extend it for steps 4–5 |
| `train_advanced.py` | Same loop but with non-scalar logging: `wandb.Image`, `wandb.Histogram`, a matplotlib figure, and `wandb.plot.confusion_matrix` | **Yes** — pick at least one to wire up for step 6 |
| `sweep.yaml` | Sweep configuration: Bayesian search over `learning_rate`, `batch_size`, `dropout` | **Yes** — try different `method` values for step 10 |
| `wandb.dockerfile` | Python 3.11 slim image, `uv pip install --system wandb`, entrypoints `train.py` | **No** — used for step 11 |
| `requirements.txt` | Lean container-only deps (used by `wandb.dockerfile`) | No |
| `pyproject.toml` | Package metadata + Python 3.11 pin + ruff/mypy config | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners that execute `train.py` then `train_advanced.py` | No — handy when reproducing the steps manually gets stuck |

## Prerequisites

1. A free W&B account at [wandb.ai/site](https://wandb.ai/site).
2. Your 40-character API key from [wandb.ai/authorize](https://wandb.ai/authorize).
3. `uv` (recommended) or plain `pip` available on your PATH. Install `uv` once:
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

## Quick start

```bash
# 1. From this folder, create a venv and install dependencies
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e .

# 2. Log in to W&B once (writes ~/.netrc on macOS/Linux, %USERPROFILE%\.netrc on Windows)
wandb login

# 3. Run the basic loop and watch the run appear in your W&B Workspace
python train.py

# 4. Run the advanced loop to log images / histograms / matplotlib / confusion matrix
python train_advanced.py
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

## End-to-end dry run

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

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once
> per terminal session.

The demo runners do **not** call `wandb agent` — sweep agents block forever, so
sweeps are a manual step (see below).

## Running a sweep

After `wandb login`, from inside this folder:

```bash
# 1. Register the sweep — prints a sweep ID like "yourname/Week7-project/abc12345"
wandb sweep sweep.yaml

# 2. Start an agent (Ctrl+C to stop after a few runs)
wandb agent yourname/Week7-project/abc12345
```

Each agent invocation re-runs `train.py` with a different point in the search
space declared in `sweep.yaml`. Watch the Sweeps tab in the W&B UI for the
parallel-coordinates plot.

## Running in Docker

```bash
# Build (BuildKit recommended so the uv cache mount is honored)
DOCKER_BUILDKIT=1 docker build -f wandb.dockerfile -t wandb:latest .

# Run with your API key — the container authenticates automatically
docker run --rm -e WANDB_API_KEY=<your-api-key> wandb:latest

# Or load the key from a .env file:
docker run --rm --env-file .env wandb:latest
```

> **Never commit your API key.** `.env` files belong in `.gitignore`.

## Troubleshooting

- **`wandb: ERROR API key not configured`** — run `wandb login`, or pass
  `WANDB_API_KEY` in the environment.
- **Sweep agent picks the same hyperparameters every time** — confirm
  `wandb.config` values are actually used inside the training loop (e.g.,
  `lr = wandb.config.learning_rate`), not hard-coded constants.
- **`%USERPROFILE%\.netrc` is read-only on Windows** — delete the file and
  re-run `wandb login`, or run the terminal as administrator once.
- **MNIST download is slow / fails** — the loop uses `torchvision.datasets.MNIST`
  with `download=True`; the first run downloads ~10 MB into `./data/`.
