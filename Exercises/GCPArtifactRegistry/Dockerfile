# syntax=docker/dockerfile:1.7
# Python 3.11 is the course-wide default. Slim + bookworm keep the image small
# without losing the build tools we need for some scikit-learn wheels on ARM.
FROM python:3.11-slim-bookworm

# Install build dependencies in a single layer, then clean the apt cache so it
# does not end up in the final image. Most scikit-learn wheels are prebuilt for
# x86_64; on linux/arm64 the pip resolver falls back to a source install which
# needs a C compiler.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv inside the image. This is the same tool we use locally.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt requirements.txt
COPY main.py main.py

# Install dependencies with uv into the container's system Python (--system),
# using a BuildKit cache mount so repeat builds skip the wheel download.
# Alternative (plain pip): RUN pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

ENTRYPOINT ["python", "-u", "main.py"]
