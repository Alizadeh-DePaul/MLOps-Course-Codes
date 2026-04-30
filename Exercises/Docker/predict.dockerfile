# -----------------------------------------------------------------------------
# Prediction image for the SE 489 Docker exercise.
#
# Build:   docker build -f predict.dockerfile . -t predict:latest
# Run (model + input mounted from host):
#   docker run --name pred --rm \
#       -v ${PWD}/models/trained_model.npy:/app/trained_model.npy \
#       -v ${PWD}/data/example_images.npy:/app/example_images.npy \
#       predict:latest /app/trained_model.npy /app/example_images.npy
# -----------------------------------------------------------------------------

FROM python:3.11-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements.txt --no-cache-dir

COPY simple_mlops/ simple_mlops/

RUN pip install . --no-deps --no-cache-dir

# The model file and input data are expected to be bind-mounted at runtime;
# the script takes their paths as CLI arguments.
ENTRYPOINT ["python", "-u", "simple_mlops/predict.py"]
