# Deployment: GCP Cloud Functions

**Course:** SE 489 (MLOps) (Week 9 / 10)

> **Heads-up (product rename):** what GCP used to call **Cloud Functions** is
> now **Cloud Run functions**. Google merged the two serverless products on
> August 21, 2024; 2nd-gen functions are now "Cloud Run functions" and run on
> the Cloud Run infrastructure. The `gcloud functions ...` command group, the
> `cloudfunctions.googleapis.com` API, and the
> `console.cloud.google.com/functions` URL were all kept for backward
> compatibility, so every command in this folder still runs verbatim. What
> changed is the product name and the Console navigation. If you search
> "Cloud Functions Python" you will land on accurate, current docs.

Two HTTP functions you can deploy serverlessly, plus a tiny trainer that
produces the model the second function serves:

1. **`hello/`** — the smallest possible deployable unit: an HTTP function that
   greets the caller. Deploy it, hit the URL, watch it scale to zero when idle.
2. **`knn/`** — loads a pickled scikit-learn model from a Cloud Storage bucket
   at cold start and serves predictions over HTTP.

This folder ships everything you need end-to-end. Run the demo, read the
files, modify them.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file |
| `train_model.py` | Trains a tiny KNN on the iris dataset, writes `model.pkl` |
| `hello/main.py` | `hello_mlops` HTTP function (functions-framework) |
| `hello/requirements.txt` | Deploy-time deps for the hello function |
| `knn/main.py` | `knn_classifier` HTTP function: loads model from GCS, predicts |
| `knn/requirements.txt` | Deploy-time deps for the knn function |
| `pyproject.toml` | Pins Python 3.11 and the local dev deps |
| `.gitignore` | Ignores the locally-produced `model.pkl` |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: train, upload, deploy both functions, test, clean up |

## Prerequisites

You should have already finished:

1. **Setting up Google Cloud Platform** — `gcloud` installed and authenticated,
   the `mlops489` project active, and the course APIs enabled (in particular
   `cloudfunctions.googleapis.com`, `run.googleapis.com`,
   `cloudbuild.googleapis.com`, `storage.googleapis.com`).
2. **GCP Identity and access management (IAM)**.
3. **Using GCP: Data** — you know how to create a bucket and copy files into it.

If `gcloud config get-value project` does not print `mlops489`, run the
[Setting up GCP](../SettingUpGCP/) smoke test first.

## Quick start

```bash
# 0. Pick a region. us-central1 is in the free-tier region set.
export REGION=us-central1
export BUCKET=mlops489-models          # must be globally unique; change if taken
export MODEL_FILE=model.pkl

# 1. Train the model locally -> model.pkl
uv run python train_model.py           # alt: python train_model.py inside a venv

# 2. Create a bucket and upload the model.
gcloud storage buckets create gs://$BUCKET --location=$REGION
gcloud storage cp $MODEL_FILE gs://$BUCKET/$MODEL_FILE

# 3. Deploy the hello function (2nd gen / Cloud Run functions).
gcloud functions deploy hello-mlops \
    --gen2 --runtime=python311 --region=$REGION \
    --source=hello --entry-point=hello_mlops \
    --trigger-http --allow-unauthenticated

# 4. Deploy the knn function, passing the bucket + model as env vars.
gcloud functions deploy knn-classifier \
    --gen2 --runtime=python311 --region=$REGION \
    --source=knn --entry-point=knn_classifier \
    --trigger-http --allow-unauthenticated \
    --set-env-vars=BUCKET_NAME=$BUCKET,MODEL_FILE=$MODEL_FILE

# 5. Call them. The describe command prints each function's URL.
HELLO_URL=$(gcloud functions describe hello-mlops --region=$REGION --gen2 --format="value(serviceConfig.uri)")
curl "$HELLO_URL?name=MLOPS%20engineer"

KNN_URL=$(gcloud functions describe knn-classifier --region=$REGION --gen2 --format="value(serviceConfig.uri)")
curl -X POST "$KNN_URL" -H "Content-Type: application/json" \
    -d '{"input_data": "5.1,3.5,1.4,0.2"}'
```

### Alternative (plain pip, for local runs without uv)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
python train_model.py
```

> **Install `uv` once:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (macOS / Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
> (Windows).

## Run a function locally before deploying

Each function is a standard [functions-framework](https://github.com/GoogleCloudPlatform/functions-framework-python)
target, so you can run it on your laptop without deploying:

```bash
# hello
functions-framework --source=hello/main.py --target=hello_mlops --debug
curl "http://localhost:8080/?name=MLOPS%20engineer"

# knn (needs the env vars and a model.pkl reachable in the bucket)
export BUCKET_NAME=$BUCKET MODEL_FILE=model.pkl
functions-framework --source=knn/main.py --target=knn_classifier --debug
curl -X POST localhost:8080 -H "Content-Type: application/json" \
    -d '{"input_data": "5.1,3.5,1.4,0.2"}'
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

Each runner trains the model, creates a uniquely-named bucket (so it does not
collide with anything you have in production), uploads the model, deploys both
functions, calls each one, then **always** tears everything down — deletes both
functions and the bucket — on the way out, even if a step in the middle fails.
The exit code is non-zero iff anything failed.

End-to-end runtime is ~4–7 minutes (the two deploys dominate).

## How the pieces fit together

```
   train_model.py ──► model.pkl ──► gcloud storage cp ──► gs://<bucket>/model.pkl
                                                                  │
                                                                  │ cold start
                                                                  ▼
   HTTP request ──► knn/main.py (knn_classifier) ──► loads model ──► prediction
   HTTP request ──► hello/main.py (hello_mlops)  ──► greeting
```

## Going further (PyTorch)

Once the sklearn version works, redo it with a PyTorch model: train any small
net, save its weights, upload them, and write a function that loads the weights
and returns a prediction. The deployment flow is identical — only the
load-and-predict code inside `main.py` changes.

## Rules

- **Do not edit `pyproject.toml`** unless you mean to change the runtime deps.
- **Do not hard-code the bucket or model name** into `knn/main.py`. They are
  read from the `BUCKET_NAME` / `MODEL_FILE` environment variables so the same
  code works across buckets and deployments.
- **Do not commit `model.pkl`** — it is a build artifact, already git-ignored.
  Upload it to the bucket instead.
- **Do not commit a JSON service-account key.** Cloud Run functions use the
  function's runtime service account to read the bucket; no key file needed.

## Clean up

Functions and bucket storage cost money while they sit idle. After class:

```bash
gcloud functions delete hello-mlops    --region=us-central1 --gen2 --quiet
gcloud functions delete knn-classifier --region=us-central1 --gen2 --quiet
gcloud storage rm --recursive gs://$BUCKET
```

The demo runners do all of this automatically as their last step.
