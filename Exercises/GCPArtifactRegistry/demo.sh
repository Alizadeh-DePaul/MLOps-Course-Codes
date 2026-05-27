#!/usr/bin/env bash
# Exercises/GCPArtifactRegistry/demo.sh - bash end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes the
# digits-svc image via Cloud Build, lists the image, pulls it back locally,
# and ALWAYS deletes the repository at the end (even on failure).
#
# Run from inside Exercises/GCPArtifactRegistry/ with gcloud already
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
REPO="mlops489-ar-demo-${SUFFIX}"
IMAGE="digits-svc"
TAG="v1"
AR_HOST="${REGION}-docker.pkg.dev"
IMAGE_URI="${AR_HOST}/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "Project:   ${PROJECT_ID}"
echo "Region:    ${REGION}"
echo "Repo:      ${REPO}  (will be deleted at the end)"
echo "Image:     ${IMAGE_URI}"
echo

# Cleanup runs no matter what (success, error, or Ctrl-C).
cleanup() {
    echo
    echo "---- Cleaning up repository ${REPO} ----"
    gcloud artifacts repositories delete "${REPO}" \
        --location="${REGION}" \
        --quiet || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Create the Artifact Registry repository
# ---------------------------------------------------------------------------
echo "---- Creating Artifact Registry repository ----"
gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="SE 489 demo (auto-created, auto-deleted)"

# ---------------------------------------------------------------------------
# 3. Submit a Cloud Build that pushes the image to the new repo
# ---------------------------------------------------------------------------
echo
echo "---- Submitting Cloud Build (this is the slow step) ----"
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPO=${REPO},_IMAGE=${IMAGE},_TAG=${TAG}"

# ---------------------------------------------------------------------------
# 4. List the image in Artifact Registry
# ---------------------------------------------------------------------------
echo
echo "---- Listing images in the repository ----"
gcloud artifacts docker images list \
    "${AR_HOST}/${PROJECT_ID}/${REPO}" \
    --include-tags

# ---------------------------------------------------------------------------
# 5. Configure Docker for the regional Artifact Registry host and pull
# ---------------------------------------------------------------------------
echo
echo "---- Configuring local Docker auth and pulling the image ----"
gcloud auth configure-docker "${AR_HOST}" --quiet
docker pull "${IMAGE_URI}"

echo
echo "Done. Repository will be deleted; local Docker image is on disk (docker rmi to remove)."
