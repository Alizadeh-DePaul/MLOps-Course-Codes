# Using GCP: Data (Cloud Storage as a DVC remote)

**Course:** SE 489 — MLOps (Week 9 / 10)

This is a **hands-on cloud** exercise. You will create a Google Cloud Storage
bucket with Object Versioning turned on, switch the DVC remote you set up in
the earlier `DataVersionControl` exercise from Google Drive to that bucket,
push and pull a small dataset, then prove that `version_aware` is doing real
work by time-travelling between two versions of the data.

The scripts in this folder run that whole flow end-to-end and **always clean
up the bucket at the end** (even on failure) so you don't burn credits
between class sessions.

## Prerequisites

You should already have completed:

- **Setting up Google Cloud Platform** — `gcloud` installed and
  authenticated, the `mlops489` project active, the eight course APIs
  enabled (in particular `storage.googleapis.com`).
- **GCP Identity and access management (IAM)** — you know what a principal,
  role, and policy binding are; you know why JSON service-account keys are
  discouraged in 2026.
- **Data Version Control** — you have a working `dvc` install and you have
  used a remote (Google Drive) at least once.

If the smoke test in `Exercises/SettingUpGCP/` doesn't fully pass, fix that
first before running anything here.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file — quick reference for the bucket + DVC remote commands |
| `pyproject.toml` | Python 3.11 pin, `dvc[gs]` extra so a `uv sync` gets you both DVC and the GCS support library |
| `data/sample_cars.csv` | Tiny starter dataset (~20 rows) — same one used in the `DataVersionControl` exercise so you can carry over |
| `.dvcignore` | Standard ignore template for DVC-tracked folders |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runner: create bucket, configure remote, push v1, push v2, time-travel, delete bucket |

## Quick start

```bash
uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .
```

> **Install `uv` once:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
> (macOS / Linux) or `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
> (Windows). `uv` is the course-wide default package manager.

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

All three runners use a uniquely-suffixed bucket name (e.g.
`mlops489-dvc-<8-char-suffix>`) so you can run them more than once without a
naming clash, and they `gcloud storage rm -r` the bucket at the end even on
error.

## Quick reference — the commands the demo runs

You only need these if you're stepping through by hand instead of running
the demo. Replace `<your-bucket>` with whatever bucket name you choose.

### Create a versioned bucket in a free-tier region

```bash
# Always-free 5 GB tier is only in us-west1, us-central1, us-east1.
gcloud storage buckets create gs://<your-bucket> \
    --location=us-central1 \
    --uniform-bucket-level-access

# Turn on Object Versioning (required for dvc version_aware)
gcloud storage buckets update gs://<your-bucket> --versioning
```

### Point DVC at the bucket

```bash
dvc remote add -d storage gs://<your-bucket>
dvc remote modify storage version_aware true

git add .dvc/config
git commit -m "Switch DVC remote from Google Drive to GCS"
```

### Push, pull, and time-travel

```bash
dvc add data
git add data.dvc data/.gitignore
git commit -m "Track sample_cars.csv with DVC"
dvc push

# Make a change, push v2
echo "Ford,F-150,1995,16.0,8,205,4500,USA" >> data/sample_cars.csv
dvc add data
git commit -am "Add 1990s pickup (v2)"
dvc push

# Go back to v1
git checkout HEAD~1 -- data.dvc
dvc checkout
wc -l data/sample_cars.csv          # back to v1 row count
```

### List bucket contents (modern CLI)

```bash
gcloud storage ls gs://<your-bucket>            # modern
gsutil ls         gs://<your-bucket>            # legacy (still works via shim, ~80% slower)
```

### Clean up

```bash
gcloud storage rm -r gs://<your-bucket>
```

## Rules of the game

1. **Never commit a service-account JSON key file** (`*-service-account.json`,
   `*-sa.json`, `key.json`, etc.). Local development uses ADC
   (`gcloud auth application-default login`), which writes credentials to
   `~/.config/gcloud/` — not into your repo. The repo's top-level
   `.gitignore` already excludes the common filename patterns.
2. **Don't commit the `.dvc/cache/` or `.dvc/tmp/` directories.** Both are
   git-ignored. Only `.dvc/config` (the remote configuration) should be in
   git.
3. **Always run the cleanup step.** A small bucket costs almost nothing per
   month, but a forgotten 100 GB bucket from a previous semester adds up.
   `gcloud storage ls` at the end of every working session and remove what
   you don't need.

## Reference docs

- [Cloud Storage docs](https://cloud.google.com/storage/docs)
- [Transition from gsutil to gcloud storage](https://docs.cloud.google.com/storage/docs/gsutil-transition-to-gcloud)
- [DVC Google Cloud Storage remote](https://doc.dvc.org/user-guide/data-management/remote-storage/google-cloud-storage)
- [DVC Cloud Versioning](https://dvc.org/doc/user-guide/data-management/cloud-versioning)
- [Use Object Versioning](https://docs.cloud.google.com/storage/docs/using-object-versioning)
