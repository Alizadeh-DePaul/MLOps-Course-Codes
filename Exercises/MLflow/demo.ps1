# Exercises/MLflow/demo.ps1 - Windows PowerShell end-to-end runner for the MLflow exercise.
# Run from inside Exercises/MLflow/ with a clean working tree.
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What this does (mirrors the exercise page steps 1-7 + 11):
#   1. Creates a venv and installs deps
#   2. Runs the basic single-run script (logs params + metric + model)
#   3. Runs the advanced multi-run script (logs figure, text, CSV artifacts)
#   4. Runs the MLflow Project entrypoint (uses python_env.yaml inside)
#   5. Runs the Optuna hyperparameter sweep with nested runs
#
# This script does NOT start `mlflow ui` for you. Open a separate terminal
# and run `mlflow ui` to browse what these steps logged.
$ErrorActionPreference = 'Stop'

# --- 1. Environment --------------------------------------------------------
# Install uv once: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv                                    # alt: python -m venv .venv
. .\.venv\Scripts\Activate.ps1
uv pip install -e .                        # alt: pip install -e .

# --- 2. Basic single run ---------------------------------------------------
# Creates .\mlruns\, logs one run under experiment "iris-classification-week7".
python mlflow_basic.py

# --- 3. Advanced multi-run with artifacts ----------------------------------
# Three nested runs under experiment "iris-classification-advanced-week7",
# each logging a confusion matrix PNG, a classification report TXT, and a
# CSV sample of the test set.
python mlflow_advanced.py

# --- 4. MLflow Project (uses python_env.yaml) ------------------------------
# `mlflow run` reads MLproject, builds the env from python_env.yaml in a
# scratch venv, then invokes the entrypoint with the parameters we pass.
mlflow run my_mlflow_project -P C=0.5 -P max_iter=200 --env-manager virtualenv

# --- 5. Optuna sweep -------------------------------------------------------
# 20 trials over a Random Forest, parent run + 20 nested runs in MLflow.
python optuna_tuning.py
