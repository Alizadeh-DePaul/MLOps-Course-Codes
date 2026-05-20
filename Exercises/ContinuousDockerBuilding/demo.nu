#!/usr/bin/env nu
# Exercises/ContinuousDockerBuilding/demo.nu — cross-platform local smoke test.
# Run from inside Exercises/ContinuousDockerBuilding/ with Docker Desktop
# (or a Docker daemon) already running.
#
# What this does (mirrors the local-validation half of the exercise page):
#   1. Builds the image locally with BuildKit
#   2. Runs the container to confirm the entrypoint prints its banner
#
# This script does NOT push to Docker Hub. Pushing happens automatically via
# the .github/workflows/docker-publish.yaml workflow on a real `git push`.
$env.config.error_style = "fancy"

# --- 1. Build the image ---------------------------------------------------
# -f points docker at the Dockerfile in this folder. The image is tagged
# `cdb:latest` (cdb = Continuous Docker Building). BuildKit is enabled by
# default in Docker 23.0+, so the cache mount in the Dockerfile just works.
docker build -f Dockerfile . -t cdb:latest

# --- 2. Run the container -------------------------------------------------
# --rm cleans up the stopped container after exit so re-running this script
# doesn't pile up dead containers. The entrypoint is `python -u app.py`.
docker run --rm cdb:latest
