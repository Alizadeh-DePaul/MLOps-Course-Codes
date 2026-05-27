# Setting up Google Cloud Platform

**Course:** SE 489 — MLOps (Week 9 / 10)

This is a **setup** exercise, not a coding exercise. There are no Python files
to edit here. Follow the exercise page to create your GCP account, install the
`gcloud` CLI, create the `mlops489` project, and enable the APIs you'll need
for the rest of Week 9 / 10.

Once you've finished the setup, run one of the verify scripts in this folder
as a smoke test. It checks that `gcloud` is on your PATH, that you're
authenticated, that the active project is set, and that the required APIs are
enabled. If anything is missing, it tells you which step to repeat.

## Files

| File | What it does |
| --- | --- |
| `README.md` | This file — quick reference for the gcloud commands |
| `demo.nu` / `demo.sh` / `demo.ps1` | Read-only smoke test that verifies your setup is complete |

## Quick reference — gcloud commands

You only run these once per machine. The exercise page has the narrative;
this section is for copy-paste.

### Install `uv` (one time, used by every exercise in this course)

```bash
# macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Authenticate and select your project

```bash
gcloud auth login                                       # user account, for gcloud commands
gcloud auth application-default login                   # ADC, for client libraries
gcloud config set project <your-project-id>
gcloud auth application-default set-quota-project <your-project-id>
```

> Find your project ID in the GCP Console under **Project Info**, or run
> `gcloud projects list`.

### Install the Google Cloud Python client library

```bash
uv pip install google-cloud-storage
```

### Alternative (plain pip)

```bash
pip install google-cloud-storage
```

> The course uses the modern **Cloud Client Libraries** (`google-cloud-*`),
> one per service. The older umbrella package `google-api-python-client` is
> kept around for APIs that don't have a Cloud Client Library yet, but you
> won't need it in this course.

### Enable the APIs the course uses

```bash
gcloud services enable \
    compute.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    cloudfunctions.googleapis.com \
    iam.googleapis.com \
    aiplatform.googleapis.com
```

> Enabling an API takes a few seconds per service. You only do this once per
> project. If you create a second project later, you'll need to enable them
> again on that project.

## Verify your setup

Pick whichever shell you already have. All three runners do the same checks
and produce the same summary.

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

The script will print a checklist. Green checks mean you're done; red crosses
tell you which command to run to fix that specific step.

## Don't have / can't install gcloud locally?

You can do this entire exercise in **Google Cloud Shell** — a free,
browser-based Linux terminal with `gcloud` pre-installed and
pre-authenticated. Open the GCP Console and click the small terminal icon
in the top-right header. Cloud Shell has a 50-hour weekly quota and a
40-minute inactivity timeout, so it's fine for tutorials but not for daily
development work.
