# syntax=docker/dockerfile:1.7
# Python 3.11 is the course-wide default. Slim + bookworm keep the image small.
# Agent Platform custom jobs accept any container image as long as it has an
# ENTRYPOINT or CMD that starts the training script.
FROM python:3.11-slim-bookworm

# Install build dependencies in a single layer, then clean the apt cache so it
# does not end up in the final image. PyTorch ships pre-built x86_64 wheels,
# so on a typical Cloud Build x86_64 worker no compilation is needed, but
# build-essential is here as a safety net for ARM builds and for any optional
# wheels that fall back to source.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv inside the image. Same tool we use locally and in sibling
# exercises (Artifact Registry, etc.).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt requirements.txt
COPY train.py train.py

# Install dependencies with uv into the container's system Python (--system),
# using a BuildKit cache mount so repeat builds skip the wheel download.
# Alternative (plain pip): RUN pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

# `-u` flushes stdout/stderr unbuffered so Agent Platform's stream-logs picks up
# every line as it is printed, not in 4 KB chunks.
ENTRYPOINT ["python", "-u", "train.py"]
