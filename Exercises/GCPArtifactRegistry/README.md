# Using GCP: Artifact Registry

**Course:** SE 489 (MLOps) (Week 9 / 10)

A simple scikit-learn digits classifier wrapped in a Docker image, with the
configuration files needed to build the image on Cloud Build and push it to
an Artifact Registry repository.

The exercise page walks through the narrative; this README is the quick
reference for what is in this folder and how to run it.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file |
| `main.py` | The sklearn digits classifier; entry point of the image |
| `pyproject.toml` | Pins Python 3.11 and the `scikit-learn` runtime dep |
| `requirements.txt` | Same deps in the form `pip` / Cloud Build understands directly |
| `Dockerfile` | Python 3.11-slim base, `uv pip install --system` with BuildKit cache |
| `cloudbuild.yaml` | Two-step build: `docker build`, then `docker push` to Artifact Registry |
| `.gitignore` | Standard Python ignores |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: create repo, build + push image, list, pull, delete |

## Prerequisites

You should have already finished:

1. **Setting up Google Cloud Platform**: `gcloud` installed and authenticated,
   `mlops489` project active, the eight course APIs enabled (in particular
   `artifactregistry.googleapis.com` and `cloudbuild.googleapis.com`).
2. **GCP Identity and access management (IAM)**.
3. **Docker** (Week 4): you should be comfortable with `docker build`,
   `docker run`, and tag syntax.

If `gcloud config get-value project` does not print `mlops489`, run the
[Setting up GCP](../SettingUpGCP/) smoke test first.

## Quick start

```bash
# 1. Pick a region and a repo name. us-central1 is in the free-tier region set.
export REGION=us-central1
export REPO=mlops489-docker

# 2. Create the Artifact Registry repository (one time).
gcloud artifacts repositories create $REPO \
    --repository-format=docker \
    --location=$REGION \
    --description="MLOps 489 Docker registry"

# 3. Submit a build. Cloud Build builds in the cloud and pushes the image
#    to the repo created above. The substitutions match cloudbuild.yaml.
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions=_REGION=$REGION,_REPO=$REPO,_IMAGE=digits-svc,_TAG=v1

# 4. Confirm the image is in the repo.
gcloud artifacts docker images list \
    $REGION-docker.pkg.dev/$(gcloud config get-value project)/$REPO

# 5. Configure Docker to authenticate against the regional Artifact Registry
#    host (one time per machine per region), then pull the image locally.
gcloud auth configure-docker $REGION-docker.pkg.dev
docker pull $REGION-docker.pkg.dev/$(gcloud config get-value project)/$REPO/digits-svc:v1
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
.\demo.ps1           # Windows PowerShell (no extra install needed)
```

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once
> per terminal session.

Each runner creates a uniquely-named repository (so it does not collide with
anything you have in production), builds and pushes one image, lists the
image, pulls it back locally, then **always deletes the repository** on the
way out, even if a step in the middle fails. The exit code is non-zero iff
anything failed.

## Rules

- **Do not edit `pyproject.toml` or `requirements.txt`.** The pinned versions
  match what the exercise page references.
- **Do edit `cloudbuild.yaml`** when the exercise tells you to. The file
  ships with default substitution values that you will override at trigger
  creation time or with `--substitutions` on the command line.
- **Do not commit a JSON service-account key** to your fork. The exercise
  page covers the keyless pattern; see also the IAM exercise.

## Clean up

Compute costs for this exercise are dominated by the storage you leave behind
in Artifact Registry. After class:

```bash
gcloud artifacts repositories delete $REPO --location=$REGION --quiet
```

The runners above do this automatically as their last step. Manual builds do
not; clean up by hand or you will keep paying for image storage.
