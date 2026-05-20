# Exercises/ContinuousDockerBuilding/demo.ps1 — Windows PowerShell local smoke test.
# Run from inside Exercises/ContinuousDockerBuilding/ with Docker Desktop
# already running.
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#
# What this does (mirrors the local-validation half of the exercise page):
#   1. Builds the image locally with BuildKit
#   2. Runs the container to confirm the entrypoint prints its banner
#
# This script does NOT push to Docker Hub. Pushing happens automatically via
# the .github\workflows\docker-publish.yaml workflow on a real `git push`.
$ErrorActionPreference = 'Stop'

# --- 1. Build the image ---------------------------------------------------
docker build -f Dockerfile . -t cdb:latest

# --- 2. Run the container -------------------------------------------------
docker run --rm cdb:latest
