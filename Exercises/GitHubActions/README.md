# Exercise: GitHub Actions (Continuous Integration)

**Course:** SE 489 - MLOps (Week 8, Continuous Integration)

This starter is a deliberately tiny Python project (`simple_mlops/calc.py`
plus four passing pytest tests). The goal is to focus on *the CI pipeline*,
not on the code under test. Two ready-to-use workflow files live under
`.github/workflows/`:

| File | What it does |
| --- | --- |
| `tests.yaml` | Runs `pytest` on Ubuntu under Python 3.11 on every push and PR to `main`. |
| `codecheck.yaml` | Runs `ruff check`, `ruff format --check`, and `mypy` on every push and PR to `main`. |

You'll extend these workflows in the sub-exercises - adding an OS matrix, a
Python-version matrix, dependency caching, and (optionally) DVC + Codecov
integration.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `simple_mlops/calc.py` | Tiny module under test (add, divide). | No |
| `tests/test_calc.py` | Four passing pytest tests. | No |
| `.github/workflows/tests.yaml` | Starter test workflow. | **Yes** (sub-exercises 6, 7) |
| `.github/workflows/codecheck.yaml` | Starter lint+typecheck workflow. | **Yes** (sub-exercise 11) |
| `pyproject.toml` | Pinned deps, pytest/coverage/ruff/mypy config. | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner (sync env, run tests, run lint, optionally run workflows locally with `act`). | No |

## Quick start

This exercise targets **Python 3.11**.

```bash
# Install uv once (skip if you have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh             # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. Sync the venv from uv.lock:
uv sync --frozen --extra dev

# 2. Activate the venv:
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 3. Confirm the tests pass locally before touching the workflows:
uv run pytest -v
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -v
```

## Running the workflows for the first time

1. Create a brand new GitHub repository (private is fine).
2. Push this folder up as the repo root. The `.github/workflows/` files are
   the only thing GitHub needs to discover your workflows.
3. Open the **Actions** tab on github.com. You should see two workflows
   listed (`Run tests`, `Code checks`) and a green check next to your push
   after a minute or so.
4. If you'd rather validate the YAML *before* you push, install
   [`act`](https://nektosact.com/) and run `act push` from the project root.
   The demo runner does this for you automatically if `act` is on `PATH`.

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

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

## Rules of the game

1. **Don't edit `simple_mlops/calc.py` or `tests/test_calc.py`** - the point
   is to get green CI, not to debug the tests.
2. Use the **`@vN` major-version pin** for actions (e.g. `actions/checkout@v5`),
   not `@vN.M.K` or `@main`. Pinning to `@main` is a security risk; pinning
   too tightly is a maintenance burden. The major-version pin is the sweet
   spot for a course exercise.
3. Trigger your workflows on **`main`** (the modern GitHub default). If your
   default branch is named differently, change `branches: [main]` to match.
4. When you finish a sub-exercise, push and confirm CI is green before
   moving on.
