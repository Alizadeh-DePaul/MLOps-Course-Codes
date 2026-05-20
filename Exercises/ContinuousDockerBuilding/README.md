# Continuous Docker Building — Publish Docker Images via GitHub Actions

**Course:** SE 489 — MLOps (Week 8, Continuous Integration)

This starter wires up GitHub Actions so every push to your repo rebuilds your
Docker image and publishes it to Docker Hub. The image itself is deliberately
minimal (a Python 3.11 base + a small `app.py` + two harmless deps) so that:

- First-time builds finish in well under a minute.
- The pip-install layer is big enough to show off BuildKit cache wins.
- You stay focused on the CI pipeline, not the application code.

Follow the exercise page for the step-by-step narrative. The files here are
what you actually edit and push.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `Dockerfile` | The image to build and publish | Maybe — try swapping in your own first |
| `app.py` | Tiny entrypoint that prints a banner | Probably not |
| `requirements.txt` | Two harmless deps (`numpy`, `rich`) so the cache layer is real | No |
| `pyproject.toml` | Package metadata | No |
| `.dockerignore` | Keeps `.git`, `__pycache__`, `.venv` out of the build context | No |
| `.github/workflows/docker-publish.yaml` | The CI workflow that builds and publishes | **Yes** — sub-steps 5, 8, 9 |
| `demo.nu` / `demo.sh` / `demo.ps1` | Local end-to-end runners — build and run the image without pushing | No — handy if you get stuck |

## Quick start

This exercise targets **Python 3.11** and assumes Docker Desktop (or a Docker
daemon) is already running.

```bash
# 1. Build the image locally so you know it works before pushing
docker build -f Dockerfile . -t cdb:latest

# 2. Run it to confirm the entrypoint prints the banner
docker run --rm cdb:latest
```

That's the local smoke test. The real work happens in CI once you wire up
the secrets on your GitHub repo and push the workflow file.

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

These runners only build and run the image locally — they do **not** push to
Docker Hub. Pushing happens automatically when GitHub Actions runs your
workflow on a real `git push`.

## What the workflow does (cheat sheet)

| Stage | Action | Purpose |
| --- | --- | --- |
| Checkout | `actions/checkout@v5` | Fetch the repo into the runner |
| Buildx | `docker/setup-buildx-action@v3` | Enable BuildKit (needed for GHA cache + multi-platform) |
| Login | `docker/login-action@v3` | Authenticate to Docker Hub with the PAT secret |
| Metadata | `docker/metadata-action@v5` | Generate consistent `:sha-<long>` and `:latest` tags |
| Build & push | `docker/build-push-action@v6` | Build the image, push it, use GHA cache |

On pull requests, the login and push steps are skipped — the build still runs
to verify the Dockerfile compiles, but no image is published. Only pushes to
`main` publish.

## BuildKit cache wins

On the first push, the workflow has nothing to cache and the dependency layer
is built from scratch. From the second push onwards, BuildKit reuses the
`pip install -r requirements.txt` layer from the GHA cache as long as
`requirements.txt` is unchanged. Expect a 5–10× speedup on subsequent runs.

To verify the cache is being used, look for a `CACHED` annotation in the
workflow's build step output.

## Multi-platform builds (optional, sub-step 9)

Adding `docker/setup-qemu-action@v3` and `platforms: linux/amd64,linux/arm64`
to the build-push step produces an image that runs on both Intel/AMD64 hosts
and ARM64 hosts (Apple Silicon, AWS Graviton, Raspberry Pi). Emulated arm64
builds on amd64 runners are 3–5× slower, so enable this only when you need
cross-platform images.

## Gotchas

- **`master` vs `main`**: the workflow triggers on `branches: [main]`. If
  your repo's default branch is still `master`, either rename it
  (`git branch -m master main`, then update the remote default branch in
  the repo's Settings) or change the trigger in the workflow.
- **Case-sensitive Docker Hub paths**: `DOCKER_HUB_USERNAME` is case-sensitive
  on Docker Hub. If the workflow fails with `unauthorized: incorrect username
  or password`, double-check the casing of the secret value.
- **First push is slow**: ~2–3 minutes with no cache. Subsequent pushes
  finish in under a minute. Patience on push #1.
- **PAT expiration**: Docker Hub PATs can expire. If a previously-working
  workflow starts failing with auth errors weeks later, generate a new PAT
  and update the `DOCKER_HUB_TOKEN` secret.
- **Apple Silicon developers**: the workflow defaults to a single
  `linux/amd64` build because GitHub-hosted runners are amd64. The resulting
  image still runs on Apple Silicon via Docker Desktop's emulation, but if
  you want native arm64 images, enable the optional multi-platform sub-step.

## Rules of the game

1. Don't commit your Docker Hub PAT to the repo. Use GitHub secrets.
2. Test the build locally with `demo.sh` (or `.nu` / `.ps1`) before pushing.
   Failed builds in CI cost minutes; failed builds locally cost seconds.
3. Don't edit `requirements.txt`, `pyproject.toml`, or `app.py` unless you're
   intentionally exercising the cache invalidation flow — the exercise is
   about the CI pipeline, not the app.
