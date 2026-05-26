# Streamlit / Gradio + Hugging Face Spaces — Bootstrap Kit

**Course:** SE 489 — MLOps (Week 9 / Week 10)

This folder is a **complete, deploy-ready bootstrap** for putting a model
behind a web UI and hosting it on Hugging Face Spaces. There's nothing to
fill in — both apps work out of the box. Pick **Streamlit** or **Gradio**,
copy the folder into your MLOps project, swap the model + class labels, and
ship.

## What's in the box

Two equivalent reference apps backed by the same tiny CIFAR-10 CNN, plus the
deployment plumbing for both.

| File | Purpose | When you'd touch it |
| --- | --- | --- |
| `app.py` | **Streamlit** reference app (tabs, batch upload, CSV export, analytics) | Swap the model bits marked `# >>> REPLACE FOR YOUR MODEL` |
| `gradio_app.py` | **Gradio** reference app (same features, Blocks API) | Same swap markers |
| `train_model.py` | Tiny CNN trainer; produces `model.pth` | Replace with your own training script |
| `pyproject.toml` | Pinned deps for both apps + dev extras (ruff, mypy, pytest) | Add your own deps |
| `huggingface_space/README.md` | HF Spaces front-matter for the **Streamlit** path | Edit `title`, `emoji`, `sdk_version` |
| `huggingface_space_gradio/README.md` | HF Spaces front-matter for the **Gradio** path | Same |
| `.github/workflows/deploy-to-hf.yml` | CI: lint, test, push to your HF Space on every commit | Set repo secrets (see below) |
| `tests/test_app.py` | Smoke tests that gate CI deploys | Add your own as you adapt |
| `demo.nu` / `demo.sh` / `demo.ps1` | End-to-end runners — pick whichever shell you have | No |
| `.streamlit/config.toml` | Local Streamlit dev-server theme | Optional |

## Streamlit vs Gradio — which one?

| | Streamlit | Gradio |
| --- | --- | --- |
| **Best for** | Dashboards, multi-page apps, anything table-heavy | Model demos, single-prompt-and-result patterns |
| **Layout control** | High (columns, tabs, sidebar, custom components) | Medium (Blocks API gives you most of what you need) |
| **Boilerplate** | Slightly more | Minimal |
| **Built-in API** | No (you'd add FastAPI yourself) | Yes — every Gradio app exposes `/api/predict` automatically |
| **HF Spaces SDK key** | `sdk: streamlit` | `sdk: gradio` |

The two apps in this folder do the same thing. Open both, decide which one
you like writing more, and delete the other. Or keep both deployed and A/B
them.

## Prerequisites

- Python 3.11 (course default)
- A free Hugging Face account: <https://huggingface.co/join>
- An HF access token with **write** scope: <https://huggingface.co/settings/tokens>

> **Install `uv` once** (course default package manager):
> `curl -LsSf https://astral.sh/uv/install.sh | sh` on macOS/Linux, or
> `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"` on Windows.

## Quick start

```bash
uv venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# 1. Train the tiny CIFAR-10 model (writes model.pth here)
python train_model.py

# 2a. Streamlit UI on http://localhost:8501
streamlit run app.py

# 2b. ...or Gradio UI on http://localhost:7860
python gradio_app.py
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python train_model.py
streamlit run app.py            # or: python gradio_app.py
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

The runners install dependencies, train the tiny model, run tests, and
launch the Streamlit app in headless mode for a 10-second smoke check.

## Deploying to Hugging Face Spaces

There are two paths. Pick one for your first deploy; sub the other in any time.

### Path A — manual git push (good for the first deploy)

```bash
# Install the HF CLI (the package also provides huggingface_hub for Python)
uv pip install -U huggingface_hub
hf auth login                  # paste your write token

# Create the Space (replace 'streamlit' with 'gradio' for the Gradio path)
hf repo create cifar10-classifier --repo-type space --space-sdk streamlit

# Push your code
git remote add hf https://huggingface.co/spaces/<your-username>/cifar10-classifier
cp huggingface_space/README.md ./README.md.hf  # or huggingface_space_gradio/README.md
git checkout -b hf-deploy
git push hf hf-deploy:main
```

### Path B — automated via GitHub Actions

`.github/workflows/deploy-to-hf.yml` lints, tests, then pushes to your HF
Space on every commit to `main`. Configure these repo secrets first
(Settings → Secrets and variables → Actions):

- `HF_USERNAME` — your Hugging Face username
- `HF_TOKEN` — the write-scope access token
- `HF_SPACE_NAME` — the Space name (e.g. `cifar10-classifier`)

The workflow copies `huggingface_space/README.md` to repo root before pushing
so HF Spaces picks up the YAML front-matter. To deploy the Gradio variant
instead, change one line in the workflow:

```yaml
- name: Stage the HF Spaces README at repo root
  run: cp huggingface_space_gradio/README.md README.md
```

## What gets deployed where

| File | Lives in your GitHub repo | Lives in your HF Space |
| --- | :---: | :---: |
| `app.py` (Streamlit) | yes | yes — if `sdk: streamlit` |
| `gradio_app.py` (Gradio) | yes | yes — if `sdk: gradio` |
| `train_model.py` | yes | no — train locally, commit `model.pth` separately |
| `tests/` | yes | no |
| `pyproject.toml` | yes | yes (HF Spaces reads it) |
| `huggingface_space/README.md` | yes (source of truth) | yes (copied to root on push) |
| `huggingface_space_gradio/README.md` | yes | only if you're using the Gradio path |
| `.github/workflows/*.yml` | yes | no |

## Adapting this to your own model

1. Open `app.py` (or `gradio_app.py`). Find the comment fences:
   ```python
   # >>> REPLACE FOR YOUR MODEL ...
   # <<< REPLACE FOR YOUR MODEL ...
   ```
2. Replace inside that block:
   - The class labels list
   - The model class definition
   - `load_model()` (point at your weights file)
   - `preprocess_image()` / the `TRANSFORM` pipeline
   - `predict()` (Streamlit) or the inference inside `predict_single` /
     `predict_batch` (Gradio)
3. Update `huggingface_space/README.md` (or `huggingface_space_gradio/README.md`)
   with your title, emoji, license.
4. Bump deps in `pyproject.toml` if your model needs them.
5. `python train_model.py` (or your equivalent) to regenerate `model.pth`.
6. Commit, push to GitHub, watch CI deploy.

If your inputs aren't images — text, audio, video, tabular — swap the
input widget (`st.file_uploader` / `gr.Image`) for the right one and adjust
the preprocessing function.

## Submitting (if this is the graded version)

Push your edits to your GitHub repo. The deploy workflow runs on every push
to `main`; a green run plus a working HF Space URL is the deliverable.
