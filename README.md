# SE 489 · MLOps — Course Code

Reference code, starter files, and exercise scaffolding for **SE 489: Machine Learning Operations (MLOps)** at DePaul University, taught by [Vahid Alizadeh](https://github.com/Alizadeh-DePaul).

> Exercise instructions, lecture content, and class notes live in the course website. This repo holds the **code** — what you'll open in your editor, run locally, and commit back. 

---

## Repository layout

```
MLOps-Course-Codes/
├── Exercises/                  ← starter code students work on (tracked)
│   ├── GoodCodingPractices/        Week 3
│   ├── Reproducibility/            Week 3   (supplementary code)
│   ├── DataVersionControl/         Week 3
│   ├── Hydra/                      Week 4
│   ├── Docker/                     Week 4
│   ├── MLCodeDebugging/            Week 5 / 6
│   ├── PerformanceProfiling/       Week 5 / 6
│   ├── ApplicationLogging/         Week 7
│   ├── WandB/                      Week 7
│   ├── MLflow/                     Week 7
│   ├── PythonUnitTesting/          Week 8
│   ├── GitHubActions/              Week 8
│   └── GCP Artifact Registry/      Week 9 / 10
├── handson-ml3/                ← supplementary notebooks from Géron's book
├── intro-to-pytorch/           ← optional PyTorch primers
├── data/                       ← small sample datasets used by exercises
└── requirements.txt            ← baseline pinned deps for the exercises
```

Each exercise folder under `Exercises/` is self-contained: a short README pointing back to its page, a `pyproject.toml` (where applicable), and the Python source files students edit. Solutions for every exercise live under `Exercises-Solutions/` **locally**, but that path is git-ignored so they don't appear on GitHub.

---

## Exercises at a glance

The table below mirrors the canonical schedule from the course's Week-by-Week page. Rows marked **_starter coming_** are exercises that are taught from the course site but don't have starter code in this repo yet.

| Week | Topic | Folder |
| :---: | --- | --- |
| 1 | IDE setup | _starter coming_ |
| 1 | Dependency Manager | _starter coming_ |
| 2 | Git & GitHub for MLOps | _starter coming_ |
| 2 | Code Structure with Cookiecutter | _starter coming_ |
| 3 | Good Coding Practices | [`Exercises/GoodCodingPractices/`](Exercises/GoodCodingPractices/) |
| 3 | Reproducibility (supplementary code) | [`Exercises/Reproducibility/`](Exercises/Reproducibility/) |
| 3 | Data Version Control | [`Exercises/DataVersionControl/`](Exercises/DataVersionControl/) |
| 4 | Hydra (Configuration Management) | [`Exercises/Hydra/`](Exercises/Hydra/) |
| 4 | Docker | [`Exercises/Docker/`](Exercises/Docker/) |
| 5 / 6 | ML Code Debugging | [`Exercises/MLCodeDebugging/`](Exercises/MLCodeDebugging/) |
| 5 / 6 | Python and ML Code Performance Profiling | [`Exercises/PerformanceProfiling/`](Exercises/PerformanceProfiling/) |
| 7 | Application Logging in Python | [`Exercises/ApplicationLogging/`](Exercises/ApplicationLogging/) |
| 7 | Experiment Tracking with WandB | [`Exercises/WandB/`](Exercises/WandB/) |
| 7 | Experiment Tracking with MLflow | [`Exercises/MLflow/`](Exercises/MLflow/) |
| 8 | Python Unit Testing | [`Exercises/PythonUnitTesting/`](Exercises/PythonUnitTesting/) |
| 8 | GitHub Actions (CI) | [`Exercises/GitHubActions/`](Exercises/GitHubActions/) |
| 8 | Pre-commit | _starter coming_ |
| 8 | Continuous Docker Building | _starter coming_ |
| 9 / 10 | Continuous Machine Learning (CML) | _starter coming_ |
| 9 / 10 | Streamlit UI and HuggingFace Deployment | _starter coming_ |
| 9 / 10 | Setting up Google Cloud Platform | _starter coming_ |
| 9 / 10 | GCP Identity and Access Management (IAM) | _starter coming_ |
| 9 / 10 | Using GCP: Compute Engine | _starter coming_ |
| 9 / 10 | Using GCP: Data | _starter coming_ |
| 9 / 10 | Using GCP: Artifact Registry | [`Exercises/GCP Artifact Registry/`](Exercises/GCP%20Artifact%20Registry/) |
| 9 / 10 | Using GCP: Training Models | _starter coming_ |
| 9 / 10 | API and Requests | _starter coming_ |
| 9 / 10 | FastAPI Application | _starter coming_ |
| 9 / 10 | Deployment: GCP Cloud Functions | _starter coming_ |
| 9 / 10 | Deployment: GCP Cloud Run | _starter coming_ |

---

## Getting started

```bash
# 1. Clone
git clone https://github.com/Alizadeh-DePaul/MLOps-Course-Codes.git
cd MLOps-Course-Codes

# 2. Create an isolated environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install baseline deps (each exercise may add more)
pip install -r requirements.txt

# 4. Jump into whichever exercise the lecture points to
cd Exercises/GoodCodingPractices
cat README.md
```

Most exercises only touch files inside their own folder, so you can safely work on one without worrying about the rest.

---

## Conventions

- **Python 3.11** is the target for new exercises added in 2026. Older folders may still target 3.9/3.10.
- **Ruff** (linter + formatter) and **mypy** (type checker) are the default code-quality tools. Exercises that use them ship a `pyproject.toml` with the relevant config.
- Starter files contain `# TODO:` comments where students should make changes. The docstring for each file explains the goal.
- Solutions live in `Exercises-Solutions/<Exercise>/` **locally** and are deliberately excluded from git.

---

## Contributing / feedback

Found a typo or a bug in the starter code? Open an issue or a PR — or just flag it in class. Suggestions that make the exercises clearer or more realistic are always welcome.

---

*Maintained by Vahid Alizadeh · DePaul University · School of Computing*
