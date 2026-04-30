# -----------------------------------------------------------------------------
# Training image for the SE 489 Docker exercise.
#
# This file matches the state described in step 12 of the exercise page:
# pip installs use --no-cache-dir to keep the image small. In step 14 you
# will REPLACE the first `pip install -r requirements.txt --no-cache-dir`
# line with a BuildKit cache mount to speed up rebuilds.
#
# Build:   docker build -f train.dockerfile . -t train:latest
# Run:     docker run --name exp1 -v ${PWD}/models:/app/models train:latest
# -----------------------------------------------------------------------------

# 1. Start from a slim Python base. Bookworm = Debian 12 (current LTS).
FROM python:3.11-slim-bookworm

# 2. System packages needed to build some wheels. Clean up afterwards to keep
#    the layer small.
RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Copy metadata first so requirement changes bust the cache, but code
#    changes do not. --no-cache-dir keeps pip from baking its download cache
#    into the image layer.
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir

# 4. Copy the project code. Order matters: changes here won't invalidate the
#    dependency layer above.
COPY simple_mlops/ simple_mlops/

RUN pip install . --no-deps --no-cache-dir

# 5. Use -u so print() statements flush to `docker logs` in real time.
ENTRYPOINT ["python", "-u", "simple_mlops/train.py"]
