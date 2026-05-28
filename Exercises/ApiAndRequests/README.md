# API and Requests — Starter

**Course:** SE 489 — MLOps (Week 9 / Week 10)

A hands-on introduction to HTTP, the `requests` package, and `curl`. You'll
practice the four verbs you'll use in every model-serving stack (`GET`,
`POST`, plus headers and binary payloads) before the next exercise wraps
your model in FastAPI.

## What's in the folder

| File | What it teaches |
| --- | --- |
| `step1_status_codes.py` | Sending a `GET`, reading `status_code`, branching on `200` vs `404`. |
| `step2_payloads.py` | `.content` vs `.json()`, GitHub Search API with `params=`. |
| `step3_binary_download.py` | Downloading a PNG — why `.json()` won't work and how to write bytes. |
| `step4_post_form_vs_json.py` | `POST` with `data=` (form-encoded) vs `json=` — the #1 integration bug. |
| `step5_curl_equivalents.md` | Reference table mapping each Python call to its `curl` equivalent. |
| `pyproject.toml` | Pinned deps + dev extras (`ruff`, `mypy`, `pytest`). |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have. |

Each `step*.py` file is self-contained. Open it, find the `# TODO:` markers,
and fill them in. The module docstring at the top explains the goal.

## Prerequisites

- Python 3.11 (course default)
- An internet connection (you'll hit `api.github.com`, `raw.githubusercontent.com`,
  and `httpbin.org`)

> **Install `uv` once** (course default package manager):
> `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux, or
> `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows.

## Quick start

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

python step1_status_codes.py
python step2_payloads.py
python step3_binary_download.py
python step4_post_form_vs_json.py
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python step1_status_codes.py
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

The runners create a venv, install dependencies, and execute the four
Python steps in order. The binary-download step writes `img.png` into this
folder — already covered by `.gitignore`.

## Gotchas you'll hit

- **GitHub unauthenticated rate limit is 60 req/hour, and the search
  endpoint is even lower (10 req/min).** If `response.status_code` is
  `403` with a `X-RateLimit-Remaining: 0` header, you've hit it — wait
  an hour or attach a token.
- **`response.json()` raises on non-JSON responses.** `requests.get("https://github.com/...")`
  returns `text/html`, not JSON, so calling `.json()` will throw a
  `JSONDecodeError`. Always check `response.headers["Content-Type"]` first
  if you're not sure.
- **`data=` sends form-encoded, `json=` sends JSON.** Step 4 makes you
  feel this difference by sending the same payload both ways and reading
  the echo. Pick the wrong one and your API thinks you sent garbage.
- **PowerShell aliases `curl` to `Invoke-WebRequest`** — a completely
  different tool with different flags. On Windows, type `curl.exe` to
  force the real curl.

## Adapting this to your own project

After the FastAPI exercise next session, you'll use these same patterns to
hit *your own* model server:

- `requests.get(API + "/health")` for readiness checks.
- `requests.post(API + "/predict", json={"image_b64": ...})` for inference.
- `curl` for one-off command-line testing during deployment.

So the takeaway from this exercise is not "memorize the `requests` API" —
it's "understand the HTTP contract well enough that you can debug any client
library against any server."
