# Docker — Containerize a Minimal Training & Prediction Pipeline

**Course:** SE 489 — MLOps (Week 4, Reproducibility in MLOps)

This package is the starter scaffold for the Docker exercise. You'll build two
images — one that *trains* a dummy model, and one that *uses* it for prediction —
so the focus stays on Docker mechanics (images, containers, volumes, build cache)
rather than on ML specifics.

Follow the exercise page for the step-by-step narrative. The files here are what
you actually edit and build.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `train.dockerfile` | Dockerfile for the training image | **Yes** — step 12 starter; you'll modify it in step 14 |
| `predict.dockerfile` | Dockerfile for the prediction image | **Yes** — similar pattern, different entrypoint |
| `requirements.txt` | Minimal runtime deps (numpy) | No |
| `pyproject.toml` | Package metadata so `pip install .` works | No |
| `.dockerignore` | Keeps junk out of the build context | No |
| `simple_mlops/train.py` | Tiny script: generates a "model" (numpy weights) and saves it | No — read it, but don't edit for this exercise |
| `simple_mlops/predict.py` | Tiny script: loads the model + some input and prints a prediction | No |
| `data/make_example_data.py` | Generates `example_images.npy` for the predict image | No |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have | No — handy if you get stuck reproducing the steps |

The goal isn't to train a real model — it's to get comfortable with the
container lifecycle.

## Quick start

```bash
# 1. From this folder, generate sample input for the predict step
python data/make_example_data.py

# 2. Build the train image
docker build -f train.dockerfile . -t train:latest

# 3. Run training. The -v mount gets the saved model back onto your host.
#    PowerShell / Linux / macOS:
docker run --name exp1 -v ${PWD}/models:/app/models train:latest
#    Windows CMD:
docker run --name exp1 -v %cd%/models:/app/models train:latest

# 4. Build the predict image
docker build -f predict.dockerfile . -t predict:latest

# 5. Run prediction with mounted model + data files
docker run --name pred --rm \
    -v ${PWD}/models/trained_model.npy:/app/trained_model.npy \
    -v ${PWD}/data/example_images.npy:/app/example_images.npy \
    predict:latest /app/trained_model.npy /app/example_images.npy
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

## BuildKit cache (optional, step 14)

After you finish step 12 your `train.dockerfile` will install dependencies
with the simple form:

```dockerfile
RUN pip install -r requirements.txt --no-cache-dir
```

In step 14 you'll *replace* that line with a BuildKit cache mount so
re-installing big libraries on rebuild becomes nearly free:

```dockerfile
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

Note that you drop `--no-cache-dir` here on purpose: pip needs to actually
write to the cache mount for the optimization to work.

BuildKit is enabled by default in Docker 23.0+ (which is effectively every
install in 2026). If you're on something older, prefix the build command
with `DOCKER_BUILDKIT=1`.

## Gotchas

- **`${PWD}` vs `%cd%` vs `$(pwd)`**: Windows CMD uses `%cd%`, PowerShell uses
  `${PWD}`, Linux/macOS shells support either `${PWD}` or `$(pwd)`. When in
  doubt, `${PWD}` is the most portable.
- **Apple Silicon (M1/M2/M3/M4)**: add `--platform linux/amd64` to `docker build`
  and `docker run` if you see "exec format error" or similar. Emulation slows
  builds down, so only use it when you actually need an amd64 image.
- **Leftover containers**: every `docker run` without `--rm` leaves a stopped
  container behind. Use `docker ps -a` to see them and `docker rm <id>` to
  clean up, or `docker system prune` to wipe the lot.
- **Files "disappear" after a run**: that's because they're created *inside*
  the container's filesystem. Either use `docker cp` or bind-mount a volume
  with `-v` — step 16 on the exercise page covers both.
- **PyCharm Community**: the Docker plugin from the Marketplace gives you the
  "Services" tool window for container/image management. Using Docker as a
  Python interpreter is still Professional-only.

## Rules of the game

1. Don't edit `requirements.txt`, `pyproject.toml`, or the `simple_mlops/`
   Python files — the exercise is about the Dockerfiles.
2. When in doubt, start a container interactively with
   `docker run -it --entrypoint sh <image>` and poke around the filesystem.
