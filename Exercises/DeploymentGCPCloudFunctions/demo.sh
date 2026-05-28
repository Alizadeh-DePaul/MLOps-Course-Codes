#!/usr/bin/env bash
# Exercises/DeploymentGCPCloudFunctions/demo.sh - bash end-to-end runner.
# Trains a KNN model, uploads it to a throwaway GCS bucket, deploys two HTTP
# Cloud Run functions (hello + knn), calls both, and ALWAYS tears everything
# down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudFunctions/ with gcloud already
# authenticated and an active project set.
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
PROJECT_ID=$(gcloud config get-value project 2>/dev/null | tr -d '[:space:]')
if [[ -z "${PROJECT_ID}" ]]; then
    echo "ERROR: No active project. Run 'gcloud config set project <id>' first."
    exit 1
fi
REGION="us-central1"
SUFFIX=$(date +%s | tail -c 7)
BUCKET="mlops489-cf-demo-${SUFFIX}"
MODEL_FILE="model.pkl"
HELLO_FN="hello-mlops-${SUFFIX}"
KNN_FN="knn-classifier-${SUFFIX}"
RUNTIME="python311"

echo "Project:   ${PROJECT_ID}"
echo "Region:    ${REGION}"
echo "Bucket:    gs://${BUCKET}  (will be deleted at the end)"
echo "Functions: ${HELLO_FN}, ${KNN_FN}  (will be deleted at the end)"
echo

# Cleanup runs no matter what (success, error, or Ctrl-C).
cleanup() {
    echo
    echo "---- Cleaning up ----"
    gcloud functions delete "${HELLO_FN}" --region="${REGION}" --gen2 --quiet || true
    gcloud functions delete "${KNN_FN}"   --region="${REGION}" --gen2 --quiet || true
    gcloud storage rm --recursive "gs://${BUCKET}" --quiet || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Environment + train the model
# ---------------------------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (or PowerShell variant on Windows)
echo "---- Setting up the environment and training the model ----"
uv venv                                     # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
uv pip install -e .                          # alt: pip install -e .
python train_model.py                        # writes model.pkl

# ---------------------------------------------------------------------------
# 3. Create a bucket and upload the model
# ---------------------------------------------------------------------------
echo
echo "---- Creating bucket and uploading the model ----"
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}"
gcloud storage cp "${MODEL_FILE}" "gs://${BUCKET}/${MODEL_FILE}"

# ---------------------------------------------------------------------------
# 4. Deploy + call the hello function
# ---------------------------------------------------------------------------
echo
echo "---- Deploying hello function (this is slow) ----"
gcloud functions deploy "${HELLO_FN}" \
    --gen2 --runtime="${RUNTIME}" --region="${REGION}" \
    --source=hello --entry-point=hello_mlops \
    --trigger-http --allow-unauthenticated
HELLO_URL=$(gcloud functions describe "${HELLO_FN}" --region="${REGION}" --gen2 \
    --format="value(serviceConfig.uri)")
echo "hello URL: ${HELLO_URL}"
curl -fsS "${HELLO_URL}?name=MLOPS%20engineer"; echo

# ---------------------------------------------------------------------------
# 5. Deploy + call the knn function
# ---------------------------------------------------------------------------
echo
echo "---- Deploying knn function (this is slow) ----"
gcloud functions deploy "${KNN_FN}" \
    --gen2 --runtime="${RUNTIME}" --region="${REGION}" \
    --source=knn --entry-point=knn_classifier \
    --trigger-http --allow-unauthenticated \
    --set-env-vars="BUCKET_NAME=${BUCKET},MODEL_FILE=${MODEL_FILE}"
KNN_URL=$(gcloud functions describe "${KNN_FN}" --region="${REGION}" --gen2 \
    --format="value(serviceConfig.uri)")
echo "knn URL: ${KNN_URL}"
curl -fsS -X POST "${KNN_URL}" \
    -H "Content-Type: application/json" \
    -d '{"input_data": "5.1,3.5,1.4,0.2"}'; echo

echo
echo "Done. All cloud resources will be deleted on exit; model.pkl is left on disk."
