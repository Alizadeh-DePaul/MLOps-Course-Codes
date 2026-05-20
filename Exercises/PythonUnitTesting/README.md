# Exercise: Python Unit Testing

**Course:** SE 489 — MLOps (Week 8, Continuous Integration)

You are handed an MNIST classifier (`models/mnist_model.py`) and a
one-epoch training loop (`training/train.py`). The `tests/` folder
ships with skeleton test files that contain docstrings explaining
*what* each test should verify and `# TODO:` markers where you fill in
the body.

By the end of the exercise you should be comfortable with:

1. The pytest discovery rules — file names start with `test_`, function names start with `test_`.
2. Writing fixtures with `@pytest.fixture`.
3. Asserting shapes, sizes, value ranges, and class coverage.
4. Asserting that a function raises an expected exception with `pytest.raises`.
5. Parametrizing the same test over multiple inputs with `@pytest.mark.parametrize`.
6. Skipping tests conditionally with `@pytest.mark.skipif`.
7. Measuring code coverage with `coverage` and `pytest-cov`, and excluding files you don't care about.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `models/mnist_model.py` | The CNN classifier under test. Raises `ValueError` on malformed input. | No |
| `training/train.py` | `train_epoch(model, loader, optimizer, criterion)` — one epoch of training. | No |
| `tests/__init__.py` | Path helpers (`_PATH_DATA`) shared across the tests. | No |
| `tests/test_data.py` | Sub-exercise 4.2 — data loading tests | **Yes** |
| `tests/test_model.py` | Sub-exercises 4.3, 4.5, parametrize | **Yes** |
| `tests/test_training.py` | Sub-exercise 4.4 — training-loop test | **Yes** |
| `tests/test_error_handling.py` | Sub-exercise 4.5 (raises) + 4.7 (skipif) | **Yes** |
| `pyproject.toml` | Dependency pins, pytest config, coverage config | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner (sync env, run tests, run coverage) | No |

## Quick start

This exercise targets **Python 3.11**. Pick one install variant.

| Your machine | Use this extra |
| --- | --- |
| CPU-only laptop, Apple Silicon | `--extra cpu` |
| Windows or Linux with NVIDIA GPU | `--extra cuda` |

Don't know? Run `nvidia-smi`. If it prints a GPU table, use `cuda`. If it isn't found, use `cpu`.

```bash
# Install uv once (skip if you have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh             # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. Sync the venv with the matching torch extra:
uv sync --extra cpu                  # or --extra cuda

# 2. Activate the venv:
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 3. Run the (failing) starter tests so you see what you're filling in:
pytest -v

# 4. Fill in the TODOs, then re-run until everything passes.
```

### Alternative (plain pip)

Plain pip doesn't read `[tool.uv.sources]`, so you have to point at the
right torch index manually:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# CPU-only torch wheel:
pip install --index-url https://download.pytorch.org/whl/cpu torch
# or, for CUDA:
# pip install --index-url https://download.pytorch.org/whl/cu128 torch

pip install -e .                     # installs torchvision, pytest, pytest-cov, coverage
```

## Running tests

```bash
pytest                               # run everything
pytest tests/test_data.py -v         # run one file
pytest -v -k "shape"                 # run tests whose name contains "shape"
```

## Coverage

Two equivalent paths, pick whichever you prefer.

**Path A — pytest-cov (recommended for pytest users):**

```bash
pytest --cov=models --cov=training --cov-report=term-missing
pytest --cov=models --cov=training --cov-report=html       # writes htmlcov/
```

**Path B — standalone coverage.py (what the exercise originally taught):**

```bash
coverage run -m pytest tests/
coverage report -m
coverage html                        # writes htmlcov/
```

Either way, open `htmlcov/index.html` in a browser for a line-by-line view.

## End-to-end dry run

Three equivalent runners are provided. Pick whichever shell you prefer.

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

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the
> first time, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once per
> terminal session.

## Rules of the game

1. **Don't edit `models/mnist_model.py`, `training/train.py`, or `pyproject.toml`** unless the exercise instructions ask you to. The model and training loop are the *system under test* — you write tests *for* them, not against them.
2. Add a descriptive message to every assert (sub-exercise 4.6). A failing test should tell you *what* went wrong, not just *that* something went wrong.
3. When you finish a sub-exercise, run `pytest -v` and make sure new tests pass before moving on.
4. When you're done with everything, run the coverage command and aim to understand which lines aren't covered and why.
