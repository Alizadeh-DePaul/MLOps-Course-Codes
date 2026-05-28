# syntax=docker/dockerfile:1.7
# Container image for the SE 489 Cloud Run deployment exercise.
#
# Build:  docker build -f Dockerfile . -t cloud-run-mlops:latest
# Run:    docker run --rm -p 8080:8080 -e PORT=8080 cloud-run-mlops:latest
# Verify: open http://localhost:8080/   and   http://localhost:8080/items/1
#
# Python 3.11 is the course default; bookworm = Debian 12 (current stable). We
# never use the `latest` tag on the base image, so builds stay reproducible.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv inside the image (same tool we use locally).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

# Copy dependency metadata first so a code change doesn't bust the deps layer.
COPY requirements.txt requirements.txt

# Install into the container's system Python (--system, no venv in the image),
# with a BuildKit cache mount so repeat builds skip the wheel download.
# Plain-pip equivalent:
#   RUN pip install --no-cache-dir -r requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY basic_fastapi.py basic_fastapi.py

# Cloud Run injects the PORT env var and routes traffic to it. We default PORT
# to 8080 (Cloud Run's default) for local `docker run`, and bind uvicorn to
# 0.0.0.0:$PORT so the container "listens on the port defined by PORT" — the
# requirement Cloud Run enforces.
ENV PORT=8080
EXPOSE 8080
CMD exec uvicorn basic_fastapi:app --host 0.0.0.0 --port $PORT --workers 1
