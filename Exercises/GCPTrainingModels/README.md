# Using GCP: Training Models

**Course:** SE 489 (MLOps) (Week 9 / 10)

> **Heads-up (May 2026 rename):** what this README calls **Agent Platform**
> is the product Google previously called **Vertex AI**. Google rebranded
> it to **Gemini Enterprise Agent Platform** at Google Cloud Next '26
> (announced April 22, 2026; Console rollout completed May 21, 2026). The
> `gcloud ai` CLI command group, the `aiplatform.googleapis.com` API
> endpoint, the YAML schema, and every command in this folder were kept
> unchanged for backward compatibility — so the demo still runs verbatim.
> What changed is the product name and the GCP Console navigation. If you
> Google "Vertex AI custom job" you will land on accurate tutorials for
> this same product.

A **complete, runnable** example of training a tiny PyTorch model on Google
Cloud Platform two ways:

1. **On a Compute Engine VM** with a PyTorch Deep Learning image — clone,
   install, `python train.py`. Treats the VM like your laptop.
2. **As an Agent Platform custom job** — build the training image, push it to
   Artifact Registry, hand Agent Platform a YAML spec, watch logs stream back.

This folder ships everything you need for path #2 end-to-end. Run the demo,
read the files, modify them; there are no fill-in-the-blank TODOs.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file |
| `train.py` | Tiny PyTorch trainer: 1-layer model on synthetic data, fits in seconds, optionally writes a checkpoint to `/gcs/<bucket>/...` |
| `pyproject.toml` | Pins Python 3.11 and the `torch` runtime dep |
| `requirements.txt` | Same deps in the form `pip` / Cloud Build understands directly |
| `train.dockerfile` | Python 3.11-slim base, `uv pip install --system` with BuildKit cache, ENTRYPOINT runs `train.py` |
| `cloudbuild.yaml` | Two-step build: `docker build`, then `docker push` to Artifact Registry |
| `config_cpu.yaml` | Agent Platform CustomJob spec (one `n1-standard-4` worker, no GPU) |
| `config_gpu.yaml` | Same spec with one `NVIDIA_TESLA_T4` attached |
| `.gitignore` | Standard Python ignores |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: build + push image, submit Agent Platform job, stream logs, clean up |

## Prerequisites

You should have already finished:

1. **Setting up Google Cloud Platform** — `gcloud` installed and
   authenticated, `mlops489` project active, the eight course APIs enabled
   (in particular `compute.googleapis.com`, `artifactregistry.googleapis.com`,
   `cloudbuild.googleapis.com`, `aiplatform.googleapis.com`).
2. **GCP Identity and access management (IAM)**.
3. **Using GCP: Compute Engine** — you know how to create / SSH / delete VMs.
4. **Using GCP: Artifact Registry** — you have at least one Docker image
   pushed to a repo named `mlops489-docker` in `us-central1` (the demo
   creates a throwaway repo if you don't, but reusing the existing one is
   faster).
5. **Docker** (Week 4) — comfortable with `docker build`, `docker run`, tags.

If `gcloud config get-value project` does not print `mlops489`, run the
[Setting up GCP](../SettingUpGCP/) smoke test first.

## Quick start

```bash
# 1. Pick a region. us-central1 is in the free-tier region set and supports
#    every Agent Platform machine and GPU type we use.
export REGION=us-central1
export REPO=mlops489-docker
export IMAGE=digits-trainer
export TAG=v1

# 2. Build the training image with Cloud Build and push to Artifact Registry.
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions=_REGION=$REGION,_REPO=$REPO,_IMAGE=$IMAGE,_TAG=$TAG

# 3. Fill the imageUri placeholder in config_cpu.yaml with your project ID, or
#    use sed to do it inline (the demo runners do this for you):
PROJECT_ID=$(gcloud config get-value project)
sed "s|<project-id>|$PROJECT_ID|g" config_cpu.yaml > /tmp/config_cpu.yaml

# 4. Submit the Agent Platform custom job.
gcloud ai custom-jobs create \
    --region=$REGION \
    --display-name=mlops489-train \
    --config=/tmp/config_cpu.yaml
```

The command prints a job ID. Stream its logs:

```bash
gcloud ai custom-jobs stream-logs <job-id> --region=$REGION
```

You should see `train.py`'s output appear line-by-line: loss values per
iteration, a final `"Training done."` message, then the job moves to
`SUCCEEDED`.

### Alternative (plain pip, for local runs without uv)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

> **Install `uv` once:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
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
.\demo.ps1           # Windows PowerShell (no extra install needed)
```

> **Nushell install** (one time): `winget install nushell` on Windows,
> `brew install nushell` on macOS, or `cargo install nu` anywhere.

> **PowerShell execution policy**: if Windows blocks `.\demo.ps1` the first
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

Each runner creates a uniquely-named Artifact Registry repo (so it does not
collide with anything you have in production), builds and pushes the training
image, submits an Agent Platform custom job, streams its logs to your terminal,
then **always** cleans up — cancels any still-running job and deletes the
throwaway repo — on the way out, even if a step in the middle fails. The
exit code is non-zero iff anything failed.

End-to-end runtime is ~5–8 minutes (Cloud Build dominates).

## How the pieces fit together

```
        ┌────────────────────────┐
        │     train.dockerfile   │
        │  ┌──────────────────┐  │
        │  │   train.py       │  │   ← writes loss + checkpoint
        │  └──────────────────┘  │
        └───────────┬────────────┘
                    │  cloudbuild.yaml (docker build + push)
                    ▼
        ┌────────────────────────┐
        │   Artifact Registry    │
        │  <region>-docker.pkg.dev/<project>/<repo>/digits-trainer:v1
        └───────────┬────────────┘
                    │  config_cpu.yaml ┌──────────► imageUri
                    ▼
        ┌────────────────────────┐
        │   Agent Platform Custom Job │
        │   gcloud ai custom-jobs│
        │   create --config=...  │
        └───────────┬────────────┘
                    │  stream-logs
                    ▼
              your terminal
```

## Reading data from Cloud Storage

`train.py` supports an optional `--gcs-checkpoint-dir` flag. When the script
runs **inside an Agent Platform custom job**, Agent Platform auto-mounts your project's
GCS buckets at `/gcs/<bucket-name>/`. So this works without any GCS client
code:

```bash
python train.py --gcs-checkpoint-dir /gcs/my-bucket/checkpoints
```

The script opens the path with the standard `pathlib` and `torch.save` APIs;
the FUSE mount handles the round-trip to GCS for you. Pre-create the bucket
with the [Using GCP: Data exercise](../UsingGCPData/) flow, or skip the flag
entirely to write to local disk.

## Rules

- **Do not edit `pyproject.toml` or `requirements.txt`** unless you mean to
  change the runtime deps in the image. The pinned versions match what
  Agent Platform's pre-built containers ship.
- **Do edit `config_cpu.yaml` / `config_gpu.yaml`** — at minimum the
  `<project-id>` placeholder in `imageUri` needs your real project ID.
  The demo runners do this substitution for you in a tmp copy; for the
  manual quick start above use `sed` or your editor.
- **Do not commit a JSON service-account key** to your fork. If you need
  the keyless attached-SA pattern, the exercise page and the IAM exercise
  cover it.
- **Do not run the GPU spec without GPU quota.** Agent Platform will fail the
  job submission with a quota error if you don't have at least 1 T4 in the
  region. See the IAM exercise for the quota request flow.

## Clean up

Compute costs for this exercise are dominated by (a) any Compute Engine VMs
you start in Section 1 and forget about, and (b) the storage of training
images in Artifact Registry. After class:

```bash
# Any Agent Platform jobs still running?
gcloud ai custom-jobs list --region=us-central1 \
    --filter="state:JOB_STATE_RUNNING OR state:JOB_STATE_PENDING"

# Any Compute Engine VMs still up?
gcloud compute instances list

# Delete the throwaway Artifact Registry repo if the demo created one
# (the demo deletes it automatically; manual builds do not)
gcloud artifacts repositories list --location=us-central1
gcloud artifacts repositories delete <repo-name> --location=us-central1 --quiet
```

The demo runners do all of this automatically as their last step.
