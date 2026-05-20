# -----------------------------------------------------------------------------
# Continuous Docker Building exercise — minimal image
#
# Deliberately small so first-time CI builds finish in under a minute, but big
# enough (numpy is ~25MB on disk) that the BuildKit cache layer is observable.
#
# Build:   docker build -f Dockerfile . -t cdb:latest
# Run:     docker run --rm cdb:latest
# -----------------------------------------------------------------------------

# Pin Python minor + Debian release + slim variant for reproducibility.
# 3.11 = current course default, slim = no docs/build-tools, bookworm = Debian 12.
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy dependency manifest first so unrelated code changes don't bust the
# (cached) pip-install layer. This is the layer the GHA cache will reuse.
COPY requirements.txt requirements.txt

# Use a BuildKit cache mount so re-installing on rebuild is nearly free.
# Drop --no-cache-dir on purpose: pip needs to actually write to the mount
# for the optimization to work.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the app last — code changes only invalidate this layer, not the
# (much heavier) dependency layer above.
COPY app.py app.py
COPY pyproject.toml pyproject.toml

# -u so print() flushes to `docker logs` in real time.
ENTRYPOINT ["python", "-u", "app.py"]
