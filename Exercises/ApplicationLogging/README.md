# Application Logging in Python

Hands-on exercise for Python's built-in `logging` module: log levels,
`dictConfig`, rotating file handlers, `rich` for colorized terminal output,
and (optionally) hooking the same `dictConfig` schema into Hydra's
`job_logging` override.

Follow the exercise instructions for the step-by-step narrative. The files
here are the runnable reference you should clone and play with.

## Files

| File | What it shows | Run order |
| --- | --- | --- |
| `logger_test.py` | The five log levels with `basicConfig`. | 1 |
| `logging_to_file.py` | Cheapest possible file logger via `basicConfig(filename=...)`. | 2 |
| `logger_advanced.py` | `dictConfig` with rotating `info.log` + `error.log` file handlers. | 3 |
| `logging_rich.py` | Same `dictConfig` but with `RichHandler` for colorized console output. | 4 |
| `logging_Programmatic.py` | Same end-state as `logger_advanced.py`, built handler-by-handler in Python code. | 5 |
| `logging_conf.py` + `logging.conf` | Same idea loaded from a ConfigParser `.ini`-style file. | 6 |
| `logging_hydra.py` + `config.yaml` | The same logging dictConfig schema, but injected via Hydra's `hydra.job_logging` override. | 7 (optional) |
| `logging_pytorch.py`, `logging_tensorflow.py` | Tiny bonus snippets showing how to thread the logger through a real ML training loop. | reference |
| `pyproject.toml` | Dependency pins (`rich`, `hydra-core`, `omegaconf`). | --- |

All files write to a local `logs/` folder (auto-created on first run).

## Quick start

This exercise uses **Python 3.11**.

```bash
# Install uv once (if you don't have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh            # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex" # Windows

uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -e .

# Run the demos in order:
python logger_test.py
python logger_advanced.py
python logging_rich.py
python logging_hydra.py              # optional: Hydra integration
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

### End-to-end dry run

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

Each runs the venv setup, then `logger_test.py`, `logger_advanced.py`,
`logging_rich.py`, and `logging_hydra.py` in sequence, and then lists the
contents of `logs/` so you can see what got written.

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere. The
> `.nu` script is the preferred course-wide option because a single file
> runs identically on every OS.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

## What to look for after running

- `logs/info.log` should contain every INFO-and-above message from the
  `logger_advanced.py` and `logging_rich.py` runs, in the `detailed`
  format (level, timestamp, file:func:lineno, message).
- `logs/error.log` should contain only ERROR-and-above messages.
- When you re-run enough times to push `info.log` past 10 MB, the
  `RotatingFileHandler` should rotate it to `info.log.1`, `info.log.2`,
  etc., keeping up to 10 backups (this is what `maxBytes=10485760,
  backupCount=10` configures).
- The `logging_rich.py` console output should be colorized and include
  rich-formatted tracebacks for the simulated exception.
- After running `logging_hydra.py`, look under `outputs/<date>/<time>/`
  for a Hydra-managed `main.log` populated by the `hydra.job_logging`
  override in `config.yaml`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'rich'`** — your venv isn't
  activated, or `uv pip install -e .` wasn't run. Re-do the Quick start.
- **`logs/` is empty** — check you actually called `dictConfig(...)`
  before logging. `dictConfig` builds the handlers; without it the
  default `WARNING`-level stderr handler is all you get.
- **`disable_existing_loggers`** — if loggers from other modules go
  silent after `dictConfig(...)`, set `"disable_existing_loggers": False`
  in your dict (the starter dicts here do this).
