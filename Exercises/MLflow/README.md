# MLflow — Mastering Experiment Tracking

**Course:** SE 489 — MLOps (Week 7, Experiment Tracking)

This package is the starter scaffold for the MLflow exercise. You will track
experiments locally, log models and artifacts, run a packaged MLflow Project,
register a model, sweep hyperparameters with Optuna, and (optionally) build a
serving Docker image.

Follow the exercise page for the step-by-step narrative. The files here are
what you actually run.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `mlflow_basic.py` | First end-to-end run: log params, metric, model | **Yes** — try a different `solver` / `C` and re-run |
| `mlflow_advanced.py` | Multi-run sweep with figure + text + CSV artifacts | **Yes** — try a different hyperparameter grid |
| `register_model.py` | Promote a run's model into the Model Registry | No — pass a `RUN_ID` you copied from the UI |
| `use_model.py` | Load a registered model by name+version, predict | **Yes** — flip `model_version` after promoting v2 |
| `optuna_tuning.py` | TPE search over a Random Forest, nested MLflow runs | **Yes** — widen the search space, increase `n_trials` |
| `mlflow_docker.py` | Build a serving Docker image for a registered model | No — Docker step is optional |
| `my_mlflow_project/MLproject` | MLflow Project entrypoint definition | No |
| `my_mlflow_project/python_env.yaml` | virtualenv-based Project env | No |
| `my_mlflow_project/train.py` | The script the Project runs | No |
| `pyproject.toml` | Package metadata, Python 3.11 pin | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have | No — handy if you get stuck reproducing the steps |

The exercise is about *MLflow mechanics* — tracking, projects, registry, tuning
— not about training a great Iris classifier. The dataset is small on purpose.

## Quick start

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e .

# 1. First run — populates ./mlruns/
python mlflow_basic.py

# 2. Open the UI in another terminal (default: http://localhost:5000)
mlflow ui
```

> **Install uv once**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (macOS/Linux), or PowerShell: `irm https://astral.sh/uv/install.ps1 | iex`.

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
python mlflow_basic.py
mlflow ui
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

The runners do **not** start `mlflow ui` for you — that's a long-running
process and you want it in its own terminal so you can browse runs while the
script keeps going. Open the UI at `http://localhost:5000` after step 1.

## A note on MLflow 3.x

This exercise is on **MLflow 3.x** (the current major as of 2026). Two things
to be aware of when you read older Stack Overflow answers or blog posts:

1. **`log_model` uses `name=` instead of `artifact_path=`.** Old MLflow 2 code
   you may find online wrote `mlflow.sklearn.log_model(model, "model")` with
   `"model"` as a positional `artifact_path`. In MLflow 3 the keyword is
   `name=`: `mlflow.sklearn.log_model(model, name="model")`. The positional
   form still runs, but it emits a deprecation warning and will be removed in
   a future release. All starter files use the new form.
2. **You no longer need `with mlflow.start_run():` to log a model** — models
   are first-class entities in MLflow 3. We still wrap most steps in a run
   because it makes the *grouping* in the UI clearer, but standalone
   `log_model` calls are valid now.

## Gotchas

- **`mlflow ui` port collision**: if you already have something on port 5000
  (e.g., AirPlay receiver on macOS), pass `mlflow ui --port 5050` and update
  the tracking URI in your scripts to match.
- **`./mlruns/` is git-ignored at the course root**, so you can re-run freely
  without polluting commits. If you delete `mlruns/` you lose every run —
  that's local-tracking-only behavior. The Step 8 tracking-server section
  shows how to centralize.
- **`mlflow.models.build_docker` (Step 12)** needs Docker Desktop or a Linux
  Docker daemon up and running — start it before you run `mlflow_docker.py`.
- **Apple Silicon + Docker**: if MLflow's serving image fails to start with
  "exec format error", add `--platform linux/amd64` to the eventual
  `docker run` you use to launch the built image.

## Rules of the game

1. The exercise is about MLflow APIs and the UI — don't get pulled into
   tweaking the Iris models. Accuracy of 1.0 is fine; that's not the point.
2. Open the UI early and keep it open. Click around between every step. The
   real learning is in the UI showing you what your code logged.
