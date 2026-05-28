# Deployment: GCP Cloud Run

**Course:** SE 489 (MLOps) (Week 9 / 10)

> **Pairs with the Cloud Functions exercise.** Do
> [DeploymentGCPCloudFunctions](../DeploymentGCPCloudFunctions/) first: there you
> hand GCP a single function and it builds the container for you. Here you build
> your **own** container (your Dockerfile, your port, multiple routes) and deploy
> it to Cloud Run. Same serverless idea, one level deeper.

> **Heads-up (registry change):** Container Registry (`gcr.io/<project>/...`)
> has been shut down. Images now live in **Artifact Registry** at
> `<region>-docker.pkg.dev/<project>/<repo>/...`. The `gcr.io/cloud-builders/docker`
> *builder* image in `cloudbuild.yaml` is a different thing and still works —
> only the place your image is stored changed.

A basic FastAPI app, containerized and deployed to Cloud Run, plus a
`cloudbuild.yaml` that builds, pushes, and deploys in one shot for continuous
deployment.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file |
| `basic_fastapi.py` | Two-route FastAPI app (`/` and `/items/{id}`) |
| `requirements.txt` | App deps (fastapi, uvicorn, pydantic) |
| `pyproject.toml` | Pins Python 3.11 and the same deps for local runs |
| `Dockerfile` | Python 3.11-slim base, `uv pip install --system`, binds uvicorn to `$PORT` |
| `cloudbuild.yaml` | Build -> push to Artifact Registry -> deploy to Cloud Run |
| `.gitignore` | Standard Python ignores |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: build, push, deploy, call, clean up |

## Prerequisites

You should have already finished:

1. **Setting up Google Cloud Platform** — `gcloud` installed and authenticated,
   the `mlops489` project active, and the course APIs enabled (in particular
   `run.googleapis.com`, `cloudbuild.googleapis.com`,
   `artifactregistry.googleapis.com`).
2. **GCP Identity and access management (IAM)**.
3. **Using GCP: Artifact Registry** — you have a Docker repo (e.g.
   `mlops489-docker` in `us-central1`).
4. **Docker** (Week 4) — comfortable with `docker build`, `docker run`, tags.

If `gcloud config get-value project` does not print `mlops489`, run the
[Setting up GCP](../SettingUpGCP/) smoke test first.

## Quick start

```bash
# 0. Config
export REGION=us-central1
export REPO=mlops489-docker
export SERVICE=cloud-run-mlops
export TAG=v1
PROJECT_ID=$(gcloud config get-value project)
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/$SERVICE:$TAG

# 1. (Optional) build + run the image locally to check it listens on $PORT.
docker build -f Dockerfile . -t $SERVICE:latest
docker run --rm -p 8080:8080 -e PORT=8080 $SERVICE:latest   # open http://localhost:8080/items/1

# 2. Build in the cloud, push to Artifact Registry, and deploy to Cloud Run
#    in one Cloud Build run (see cloudbuild.yaml).
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions=_REGION=$REGION,_REPO=$REPO,_SERVICE=$SERVICE,_TAG=$TAG

# 3. Get the URL and call it.
SERVICE_URL=$(gcloud run services describe $SERVICE --region=$REGION --format="value(status.url)")
curl "$SERVICE_URL/"
curl "$SERVICE_URL/items/1"
```

### Deploy a prebuilt image directly (no Cloud Build)

```bash
gcloud run deploy $SERVICE \
    --image $IMAGE \
    --platform managed --region $REGION --allow-unauthenticated
gcloud run services list
gcloud run services describe $SERVICE --region=$REGION
```

### Alternative (plain pip, for local runs without uv)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
uvicorn basic_fastapi:app --reload
```

> **Install `uv` once:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (macOS / Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
> (Windows).

## The `$PORT` gotcha

Cloud Run sets a `PORT` environment variable and routes traffic to it; your
container **must** listen on `0.0.0.0:$PORT` or the deploy fails with *"the
user-provided container failed to start and listen on the port defined by the
PORT environment variable."* The `Dockerfile` here handles it: it defaults
`PORT` to 8080 for local runs and binds uvicorn to `--port $PORT`. Don't
hard-code a different port.

## End-to-end dry run

Three equivalent runners are provided; pick whichever shell you prefer:

```nu
nu demo.nu           # cross-platform (Windows / macOS / Linux) - recommended
```

```bash
bash demo.sh         # macOS / Linux / WSL / Git Bash
```

```powershell
.\demo.ps1           # Windows PowerShell (no extra install needed)
```

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once
> per terminal session.

Each runner creates a uniquely-named Artifact Registry repo (so it does not
collide with anything you have in production), builds + pushes + deploys in one
Cloud Build run, calls the service, then **always** deletes the Cloud Run
service and the repo on the way out, even if a step in the middle fails. The
build happens in the cloud, so you do not need a local Docker daemon for the
runners. End-to-end runtime is ~4–7 minutes (Cloud Build dominates).

## Continuous deployment

`cloudbuild.yaml` is trigger-ready: connect the repo in Cloud Build, point a
trigger at this folder, and every push rebuilds the image and redeploys the
service. The build/push/deploy steps are the same ones the runners invoke
manually.

## Rules

- **Do not edit `pyproject.toml` or `requirements.txt`** unless you mean to
  change the app's deps.
- **Do not hard-code a port** other than via `$PORT` — Cloud Run requires it.
- **Do not commit a JSON service-account key.** Cloud Build and Cloud Run use
  attached service accounts; no key file needed.

## Clean up

A deployed service and stored images cost money. After class:

```bash
gcloud run services delete $SERVICE --region=$REGION --quiet
gcloud artifacts repositories delete $REPO --location=$REGION --quiet   # only if it was a throwaway
```

The runners do this automatically as their last step.

## Going further

Want more control than serverless gives you — managing the cluster yourself?
That is Kubernetes. A good next step is [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/v2/introduction/),
which leads into building a pipeline on **Gemini Enterprise Agent Platform**
(formerly Vertex AI).
