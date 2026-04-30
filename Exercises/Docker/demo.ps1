# Exercises/Docker/demo.ps1 - Windows PowerShell end-to-end runner for the Docker exercise.
# Run from inside Exercises/Docker/ with Docker Desktop (or daemon) already running.
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What this does (mirrors the Notion exercise page steps 9-17):
#   1. Generates sample input data
#   2. Builds the train image
#   3. Runs training, saving the model back to .\models on the host
#   4. Builds the predict image
#   5. Runs prediction with model + data bind-mounted
$ErrorActionPreference = 'Stop'

# --- 1. Sample input -------------------------------------------------------
# train.py generates a dummy model; predict.py needs example_images.npy.
# Generate that on the host so the predict step has something to bind-mount.
python data\make_example_data.py

# --- 2. Build the train image ---------------------------------------------
# Tag is `train:latest`. -f points docker at our specific Dockerfile.
docker build -f train.dockerfile . -t train:latest

# --- 3. Run training ------------------------------------------------------
# -v binds host .\models to /app/models inside the container so the
# numpy weights file lands back on disk after the container exits.
# --rm removes the container so re-running this script doesn't leave
# stopped containers behind.
New-Item -ItemType Directory -Force -Path .\models | Out-Null
docker run --rm -v "${PWD}/models:/app/models" train:latest

# --- 4. Build the predict image -------------------------------------------
docker build -f predict.dockerfile . -t predict:latest

# --- 5. Run prediction ----------------------------------------------------
# Mount the trained model and the example input as files inside the
# container; the entrypoint takes their paths as CLI arguments.
docker run --rm `
    -v "${PWD}/models/trained_model.npy:/app/trained_model.npy" `
    -v "${PWD}/data/example_images.npy:/app/example_images.npy" `
    predict:latest /app/trained_model.npy /app/example_images.npy
