# Pre-commit — Automate Code Checks Before Every Commit

**Course:** SE 489 — MLOps (Week 8, Reproducibility in MLOps)

This package is the starter scaffold for the Pre-commit exercise. Pre-commit
runs your linter, formatter, and other quick checks automatically at
`git commit` time, so problems are caught before they enter the repo.

Follow the exercise page for the step-by-step narrative. The files in this
folder are what you actually edit and run.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `.pre-commit-config.yaml` | Starter config with the four default hooks | **Yes** — you add ruff and more hooks |
| `sample_code.py` | Tiny Python file with intentional issues for hooks to flag | **No** — it's the test target |
| `pyproject.toml` | Minimal ruff + mypy config so the linter hook has something to read | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have | No — handy if you get stuck reproducing the steps |

The starter `.pre-commit-config.yaml` ships with **only** the four sample-config
hooks (`trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
`check-added-large-files`). Adding the ruff hook is part of the exercise.

## Quick start

```bash
# 1. Create and activate a virtual env
uv venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# 2. Install pre-commit (with the pre-commit-uv plugin for faster installs)
uv tool install pre-commit --with pre-commit-uv

# 3. Install the hooks into .git/hooks
pre-commit install

# 4. Run every hook against every file in the repo
pre-commit run --all-files

# 5. Once you have ruff working, see what `autoupdate` does
pre-commit autoupdate
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

> **Install `uv` once**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (macOS / Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
> (Windows).

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

## What the four default hooks do

| Hook ID | What it catches |
| --- | --- |
| `trailing-whitespace` | Strips trailing spaces at end of lines |
| `end-of-file-fixer` | Ensures every file ends with exactly one newline |
| `check-yaml` | Parses YAML to make sure it's syntactically valid |
| `check-added-large-files` | Blocks accidentally committing files > 500 kB |

`sample_code.py` is intentionally malformed (trailing whitespace, no final
newline, etc.) so these hooks have something to fix.

## Gotchas

- **Hooks only see staged files**. A fresh `pre-commit install` does nothing
  visible until your next `git commit`. Use `pre-commit run --all-files` to
  check everything immediately.
- **First run is slow**. Pre-commit downloads and installs each hook into
  isolated environments under `~/.cache/pre-commit/`. Subsequent runs reuse
  these and are fast. The `pre-commit-uv` plugin speeds up the Python ones.
- **`--no-verify` is the escape hatch**, not the default**.** `git commit -m "..."
  --no-verify` skips the hooks for that commit. Save it for "the repo is on
  fire and I need to push right now" emergencies.
- **`pre-commit autoupdate` rewrites your config**. It bumps every `rev:` tag
  to the latest release. Always commit the resulting diff separately.
- **Hooks run in isolated envs, not your venv**. If a hook needs a custom
  dependency, declare it under `additional_dependencies:` in the hook entry.

## Rules of the game

1. **Start from the four-hook starter** in `.pre-commit-config.yaml`. Don't
   paste in someone else's mega-config until you understand each hook.
2. **Read the output**. Most hook failures auto-fix the file and just need
   `git add` + re-commit. The error message tells you which.
3. **Use the exercise as your ground truth** for the step order, not this
   README. The README is a quick-reference, not the lesson.
