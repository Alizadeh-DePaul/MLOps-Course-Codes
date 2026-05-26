# Exercise: Continuous Machine Learning (CML)

**Course:** SE 489 - MLOps (Week 9, Continuous Machine Learning)

This starter wires up CML (Continuous Machine Learning by iterative.ai) so
every push to `main` (and every PR) re-trains a tiny scikit-learn model and
posts a Markdown report (classification report + confusion matrix image) as
a comment on the pull request.

The model is deliberately trivial - a logistic-regression classifier on
sklearn's bundled `digits` dataset. It trains in well under a second, has no
external data download, and never reaches for GPUs. The point is the
**reporting pipeline**, not the model.

## Files

| File | What it is | Do you edit it? |
| --- | --- | --- |
| `simple_mlops/train_model_cml.py` | Trains the model, writes `classification_report.txt` and `confusion_matrix.png`. | Maybe - try swapping the model |
| `pyproject.toml` | Pinned deps (scikit-learn, matplotlib, numpy). | No |
| `.github/workflows/cml.yaml` | The CML workflow that runs on every push and PR. | **Yes** - sub-exercises 2, 3, 6 |
| `demo.nu` / `demo.sh` / `demo.ps1` | Local end-to-end runners - train + render the report without pushing. | No |

## Quick start

This exercise targets **Python 3.11**.

```bash
# Install uv once (skip if you have it):
#   curl -LsSf https://astral.sh/uv/install.sh | sh             # macOS/Linux
#   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# 1. Sync the venv from pyproject.toml:
uv venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
uv pip install -e .

# 2. Train the model and produce the report artifacts locally:
python simple_mlops/train_model_cml.py

# 3. Confirm the two output files exist:
#    - classification_report.txt
#    - confusion_matrix.png
```

### Alternative (plain pip)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e .
python simple_mlops/train_model_cml.py
```

## End-to-end dry run

Three equivalent runners are provided. Pick whichever shell you prefer.

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

These runners only train the model and write `classification_report.txt` +
`confusion_matrix.png` locally - they do **not** post a PR comment. Posting
happens automatically when GitHub Actions runs the workflow on a real
`git push`.

## Running the workflow for the first time

1. Push this folder up to a brand new GitHub repository (private is fine).
2. **Enable PR-write permissions for `GITHUB_TOKEN`**. Go to your repo's
   Settings -> Actions -> General -> Workflow permissions and select
   "Read and write permissions". This is the single most common student
   gotcha - without it, the bot can write the report but cannot post it to
   the PR.
3. Open a branch, change something tiny in `train_model_cml.py`
   (`max_iter=200` -> `max_iter=300` is enough), push, and open a PR.
4. Within ~1 minute, you should see a comment from `github-actions[bot]` on
   the PR with your classification report and confusion matrix image.

## What the workflow does (cheat sheet)

| Stage | Action | Purpose |
| --- | --- | --- |
| Checkout | `actions/checkout@v5` | Fetch the repo into the runner |
| Install uv | `astral-sh/setup-uv@v8` | Fast, reproducible Python env |
| Set up CML | `iterative/setup-cml@v2` | Provides the `cml` CLI |
| Install deps | `uv pip install --system -e .` | Install the project deps |
| Train | `python simple_mlops/train_model_cml.py` | Produces report.txt + confusion_matrix.png |
| Report | `cml comment create --publish report.md` | Posts the Markdown report as a PR comment, uploading images automatically |

## Sub-exercises (follow the exercise page)

1. **Train and produce report artifacts.** Run the model locally first so
   you understand what the workflow will run in CI.
2. **Add the workflow.** Copy `.github/workflows/cml.yaml` into your repo.
3. **Enable PR write permissions** as described above. Push and open a PR.
4. **Iterate on the report.** Add a metrics table, a header, or a "before
   vs. after" diff to `report.md` before calling `cml comment create`.
5. **(Optional) CML + DVC.** Pull the dataset from a DVC remote instead of
   bundling it. See `Exercises-Solutions/ContinuousMachineLearning/cml_with_dvc_solution.yaml`
   (instructor reference) or [cml.dev/doc/cml-with-dvc](https://cml.dev/doc/cml-with-dvc).
6. **(Optional) Update-in-place comments.** Swap `cml comment create` for
   `cml comment update` so the PR has one comment that updates as you
   iterate instead of a wall of bot replies.

## Gotchas

- **`permissions:` block is required.** GitHub's default `GITHUB_TOKEN`
  only has read access on most repos. The workflow declares
  `permissions: { contents: read, pull-requests: write }` explicitly so the
  bot can post the comment. Don't strip this.
- **No GPU on `ubuntu-latest`.** GitHub-hosted runners are CPU-only (see
  [supported runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)).
  For GPU training, look at `cml runner launch --cloud=aws` to spin up an
  EC2 instance on demand - that's a separate (and credit-card-required)
  follow-on exercise.
- **`cml comment create --publish` vs. `--publishNative`.** Use `--publish`
  for GitHub - it uploads images to a CML-managed CDN and embeds them in
  the comment. `--publishNative` uses the host platform's native upload
  (currently only on GitLab/Bitbucket).
- **Don't paste your `GITHUB_TOKEN` anywhere.** The workflow uses
  `${{ secrets.GITHUB_TOKEN }}` which GitHub auto-provides per-run. You
  never see it, never store it.

## Rules of the game

1. Don't delete the `permissions:` block from `cml.yaml`.
2. Use `actions/checkout@v5` and `iterative/setup-cml@v2` (the modern,
   maintained versions). Older `@v1` of setup-cml and `@v2` of checkout
   are deprecated.
3. Use `cml comment create` (subcommand), not `cml-send-comment`
   (deprecated hyphenated form). Same for `cml comment create --publish`,
   not the old `cml-publish`.
