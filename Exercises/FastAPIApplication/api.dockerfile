# -----------------------------------------------------------------------------
# Container image for the SE 489 FastAPI Application exercise.
#
# Build:   docker build -f api.dockerfile . -t fastapi-app:latest
# Run:     docker run --name myapi --rm -p 8000:80 fastapi-app:latest
# Verify:  open http://localhost:8000/items/1   (host 8000 -> container 80)
# -----------------------------------------------------------------------------

# Slim Python base. Bookworm = Debian 12 (current stable). We pin 3.11 (the
# course default) and never use the `latest` tag, so builds stay reproducible.
FROM python:3.11-slim-bookworm

WORKDIR /code

# Copy dependency metadata first so a code change doesn't bust the (slow)
# dependency layer.
COPY requirements.txt /code/requirements.txt

# Install deps with uv into the container's system Python (--system, so uv
# doesn't create a venv inside the image). The BuildKit cache mount keeps uv's
# download cache across rebuilds without baking it into the image layer.
# Plain-pip equivalent:
#   RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir uv && \
    uv pip install --system -r /code/requirements.txt

# Copy the application package.
COPY ./app /code/app

# Serve the app object found at app.main:app on all interfaces, port 80.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
