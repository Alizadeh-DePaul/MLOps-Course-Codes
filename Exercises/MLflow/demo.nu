#!/usr/bin/env nu
# Exercises/MLflow/demo.nu - cross-platform end-to-end runner for the MLflow exercise.
# Run from inside Exercises/MLflow/ with a clean working tree.
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
$env.config.error_style = "fancy"

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (macOS/Linux)
#                  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  (Windows)
uv venv                                    # alt: python -m venv .venv

# Nushell doesn't source activation scripts — prepend the venv bin dir to PATH
# and set VIRTUAL_ENV ourselves. Works identically on Windows/macOS/Linux.
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

uv pip install -e .                        # alt: pip install -e .

# --- 2. Basic single run ---------------------------------------------------
# Creates ./mlruns/, logs one run under experiment "iris-classification-week7".
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
