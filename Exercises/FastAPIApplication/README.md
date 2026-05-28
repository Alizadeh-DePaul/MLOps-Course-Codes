# FastAPI Application — Complete Example

**Course:** SE 489 — MLOps (Week 9 / Week 10)

This folder is a **complete, runnable reference API** — there is nothing to
fill in. Every endpoint from the exercise is implemented and correct, so you
can read `app/main.py` top to bottom, run it, and watch each concept work.
After the previous *API and Requests* exercise (the client side), this is the
server side: you'll wrap logic behind HTTP endpoints the way you'll later wrap
your own model.

## What's in the folder

| File | What it shows |
| --- | --- |
| `app/main.py` | The complete API: path/query params, enums, POST, JSON bodies, file upload + OpenCV resize. |
| `app/ml_caption.py` | **Optional** image-captioning API (VisionEncoderDecoder). Heavy deps, behind the `[ml]` extra. |
| `tests/test_api.py` | Full test suite using FastAPI's `TestClient` — no running server needed. |
| `requirements.txt` | Core deps (`fastapi[standard]`, OpenCV, Pillow) — used by the container. |
| `pyproject.toml` | Pinned deps + `dev` and `ml` extras, ruff/mypy/pytest config. |
| `api.dockerfile` | Containerizes the API on `python:3.11-slim-bookworm`. |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have. |

## The endpoints

| Method | Path | Concept |
| --- | --- | --- |
| `GET` | `/` | Root + HTTP status code (`http.HTTPStatus`). |
| `GET` | `/items/{item_id}` | Path parameter with `int` type validation. |
| `GET` | `/restric_items/{item_id}` | Path parameter restricted to an `Enum`. |
| `GET` | `/query_items` | Query parameter. |
| `POST` | `/login/` | POST with form-style params (toy in-memory store). |
| `GET` | `/text_model/` | Single string input + regex email check. |
| `POST` | `/text_model/` | JSON body (Pydantic model) + domain match. |
| `POST` | `/cv_model/` | File upload → OpenCV resize → file response. |

## Prerequisites

- Python 3.11 (course default)

> **Install `uv` once** (course default package manager):
> `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux, or
> `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows.

## Quick start

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Run the API in development mode (auto-reload) with the FastAPI CLI:
fastapi dev app/main.py        # serves on http://127.0.0.1:8000
```

Then open the interactive docs at <http://127.0.0.1:8000/docs>, click an
endpoint, hit **Try it out**, and execute it.

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
fastapi dev app/main.py
```

### Running with uvicorn directly

`fastapi dev` is a thin wrapper around uvicorn. To see the ASGI server
explicitly (and choose the port the exercise uses):

```bash
uvicorn app.main:app --reload --port 8888    # docs at http://127.0.0.1:8888/docs
```

## Tests

```bash
pytest -v
```

FastAPI's `TestClient` drives the app in-process, so the tests pass without a
running server.

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

Each runner creates a venv, installs deps, runs the tests, then boots the API
and curls two endpoints as a smoke check. The `cv_model` and `login`
endpoints write `image*.jpg` / `database.csv` into this folder at runtime —
already covered by `.gitignore`.

## Run it in a container

```bash
docker build -f api.dockerfile . -t fastapi-app:latest
docker run --name myapi --rm -p 8000:80 fastapi-app:latest
# then visit http://localhost:8000/items/1
```

The image serves `app.main:app` on port 80; `-p 8000:80` maps it to
`localhost:8000` on your machine.

## Optional: image captioning

`app/ml_caption.py` is a separate FastAPI app that captions images with a
VisionEncoderDecoder model. It needs the heavy ML extra:

```bash
uv pip install -e ".[ml]"      # alt: pip install -e ".[ml]"
fastapi dev app/ml_caption.py
# POST an image:
curl -X POST "http://127.0.0.1:8000/caption/?max_length=24" -F "data=@your_image.jpg"
```

The first request downloads the model weights (~1 GB) and is slow; later
requests are fast.

## Gotchas you'll hit

- **`fastapi dev` defaults to port 8000**, `uvicorn ... --port 8888` uses
  8888. Match the port to whichever command you ran when you open `/docs`.
- **Visiting `/items/test` returns a `422`, not a crash** — that's FastAPI
  validating the `int` type for you. Returning structured errors instead of
  500s is a feature, not a bug.
- **OpenCV's `resize` takes `(width, height)`**, not `(height, width)`.
- **PowerShell aliases `curl` to `Invoke-WebRequest`** — type `curl.exe` to
  force the real curl when testing endpoints from Windows.

## Adapting this to your own model

Swap the body of `/cv_model/` (or add a new `/predict/` endpoint) to load your
model once at module import and run inference per request. Accept input as a
Pydantic JSON body for structured data, or `UploadFile` for images/audio, and
return either JSON or a `FileResponse`. The patterns in `app/main.py` cover
every input/output shape you'll need.
