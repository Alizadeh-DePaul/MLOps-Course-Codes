# Using GCP: Compute Engine

**Course:** SE 489 — MLOps (Week 9 / 10)

This is a **hands-on cloud** exercise, not a coding exercise. There are no
Python files to edit here. Follow the exercise page to create a small VM in
your `mlops489` project, list it, SSH into it, create a second VM from a
PyTorch Deep Learning VM image family, then stop and delete both so you
don't burn credits.

The scripts in this folder run that whole flow end-to-end as a smoke test.
Use them to dry-run before class, or as a reference if you get stuck running
the steps by hand from the exercise page.

## Prerequisites

You should already have completed:

- **Setting up Google Cloud Platform** — `gcloud` installed and
  authenticated, the `mlops489` project active, and the eight course APIs
  enabled (`compute`, `storage`, `artifactregistry`, `cloudbuild`, `run`,
  `cloudfunctions`, `iam`, `aiplatform`).
- **GCP Identity and access management (IAM)** — if you want to attach a
  GPU later, your GPU quota for the chosen region should be at least 1.

If the smoke test in `Exercises/SettingUpGCP/` doesn't fully pass, fix
that first before running anything here.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file — quick reference for the `gcloud compute` commands |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: create, list, SSH, create-from-image-family, stop, delete |

## Quick reference — `gcloud compute` commands

These are the commands the demo runs, broken out so you can copy/paste them
individually. Replace `<project-id>` with your real project ID
(`gcloud config get-value project` if you forgot).

### Create a free-tier e2-micro VM

```bash
gcloud compute instances create mlops489-cpu \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --image-family=debian-12 \
    --image-project=debian-cloud
```

> The free-tier `e2-micro` is free only in `us-west1`, `us-central1`, or
> `us-east1`, and only one instance per month. Outside those regions
> e2-micro is billable.

### List and SSH

```bash
gcloud compute instances list

gcloud compute ssh mlops489-cpu --zone=us-central1-a
```

> The exact SSH command is also available in the Console under each VM
> row → **More actions (kebab menu)** → **View gcloud command**.

### Create a PyTorch Deep Learning VM (with optional T4 GPU)

```bash
gcloud compute instances create mlops489-pytorch \
    --zone=us-central1-a \
    --image-family=pytorch-latest-cpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-4
```

For a GPU-enabled instance (requires GPU quota — see the IAM exercise):

```bash
gcloud compute instances create mlops489-pytorch-gpu \
    --zone=us-central1-a \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True"
```

> **GPU choice:** NVIDIA Tesla **K80** that older tutorials still reference
> was retired on Google Cloud on **May 1, 2024** — you cannot create a
> K80 VM anymore. The **T4** is now the cheapest training GPU (~$0.35/hr).
> L4 (~$0.71/hr) is newer; A100 / H100 are for serious training and need
> separate quota.

### List available Deep Learning image families

```bash
gcloud compute images list \
    --project=deeplearning-platform-release \
    --filter="family ~ pytorch" \
    --format="value(family)"
```

### List Deep Learning containers (replaces the old `gcloud container images list`)

The old `gcr.io` Container Registry was shut down on **March 18, 2025**.
The Deep Learning Containers now live in Artifact Registry:

```bash
gcloud artifacts docker images list \
    us-docker.pkg.dev/deeplearning-platform-release/gcr.io \
    --include-tags --limit=20
```

### Stop and delete (do this every time you're done)

```bash
gcloud compute instances stop mlops489-cpu --zone=us-central1-a
gcloud compute instances stop mlops489-pytorch --zone=us-central1-a

gcloud compute instances delete mlops489-cpu --zone=us-central1-a --quiet
gcloud compute instances delete mlops489-pytorch --zone=us-central1-a --quiet
```

> **Why both stop and delete?** Stopping a VM frees the CPU/RAM charge but
> the **attached persistent disk keeps billing**. Delete the VM (or its
> disk) when you're done with the exercise.

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
> time, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
> once per terminal session.

The runner creates two VMs, lists them, SSHes in, then **always** stops and
deletes them at the end (even if a step in the middle fails) so you don't
leave anything running.

## Don't have / can't install gcloud locally?

You can run every command in this exercise from **Google Cloud Shell** — a
free, browser-based Linux terminal with `gcloud` pre-installed and
pre-authenticated. Open the GCP Console and click the small terminal icon
in the top-right header. Cloud Shell has a 50-hour weekly quota and a
40-minute inactivity timeout, so it's fine for tutorials but not for daily
development work.

## Up next

`Exercises/UsingGCPData/` — turn a Google Cloud Storage bucket into your
DVC remote with `version_aware` turned on. The natural follow-up once you
have a VM that needs persistent, versioned data access.
