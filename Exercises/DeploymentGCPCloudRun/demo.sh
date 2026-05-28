#!/usr/bin/env bash
# Exercises/DeploymentGCPCloudRun/demo.sh - bash end-to-end runner.
# Creates a throwaway Artifact Registry repo, then builds + pushes + deploys the
# FastAPI container to Cloud Run via Cloud Build, calls the service, and ALWAYS
# tears everything down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudRun/ with gcloud already
# authenticated and an active project set. No local Docker needed — the image
# is built in the cloud by Cloud Build.
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
REPO="mlops489-cr-demo-${SUFFIX}"
SERVICE="cloud-run-mlops-${SUFFIX}"
TAG="v1"

echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Repo:     ${REPO}  (will be deleted at the end)"
echo "Service:  ${SERVICE}  (will be deleted at the end)"
echo

cleanup() {
    echo
    echo "---- Cleaning up ----"
    gcloud run services delete "${SERVICE}" --region="${REGION}" --quiet || true
    gcloud artifacts repositories delete "${REPO}" --location="${REGION}" --quiet || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Create a throwaway Artifact Registry repository
# ---------------------------------------------------------------------------
echo "---- Creating Artifact Registry repository ----"
gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="SE 489 Cloud Run demo (auto-created, auto-deleted)"

# ---------------------------------------------------------------------------
# 3. Build + push + deploy in one Cloud Build run
# ---------------------------------------------------------------------------
echo
echo "---- Build + push + deploy via Cloud Build (this is the slow step) ----"
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPO=${REPO},_SERVICE=${SERVICE},_TAG=${TAG}"

# ---------------------------------------------------------------------------
# 4. Call the deployed service
# ---------------------------------------------------------------------------
echo
echo "---- Calling the service ----"
SERVICE_URL=$(gcloud run services describe "${SERVICE}" --region="${REGION}" \
    --format="value(status.url)")
echo "Service URL: ${SERVICE_URL}"
curl -fsS "${SERVICE_URL}/";        echo
curl -fsS "${SERVICE_URL}/items/1"; echo

echo
echo "Done. All cloud resources will be deleted on exit."
