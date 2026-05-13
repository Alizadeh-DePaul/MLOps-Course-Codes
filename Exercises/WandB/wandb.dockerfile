# syntax=docker/dockerfile:1.7
# Exercises/WandB/wandb.dockerfile - SE 489 Week 7 WandB exercise.
#
# Builds a lean image that runs train.py inside the container. The container
# authenticates with W&B at runtime via the WANDB_API_KEY environment variable
# (set with `docker run -e WANDB_API_KEY=<key> ...` or `--env-file .env`).
#
# Build with BuildKit so the uv cache mount works:
#   DOCKER_BUILDKIT=1 docker build -f wandb.dockerfile -t wandb:latest .

FROM python:3.11-slim-bookworm

# Build-essential needed because torch wheels ship binaries but some transitive
# dependencies (e.g. wandb's gql) still occasionally need a compiler on slim.
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# uv is the course default. --system installs into the container's system Python
# instead of creating a venv — that's the conventional pattern in Docker images.
# The cache mount keeps re-builds fast without baking the cache into the image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache -r requirements.txt
# Plain-pip equivalent (use this line if you're not using BuildKit):
# RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script. `data/` will be created on first run by torchvision
# downloading MNIST, so we don't COPY it from the host.
COPY train.py .

# Sensible Python defaults inside a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENTRYPOINT ["python", "-u", "train.py"]
