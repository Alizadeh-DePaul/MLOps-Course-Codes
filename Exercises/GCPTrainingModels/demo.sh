#!/usr/bin/env bash
# Exercises/GCPTrainingModels/demo.sh - bash end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes
# the training image via Cloud Build, submits a Vertex AI custom job,
# streams its logs, and ALWAYS cleans up at the end (cancels the job if
# still running, deletes the repository) - even if a step in the middle
# fails.
#
# Run from inside Exercises/GCPTrainingModels/ with gcloud already
# authenticated and an active project set. Requires the aiplatform,
# artifactregistry, and cloudbuild APIs to be enabled.
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
REPO="mlops489-train-demo-${SUFFIX}"
IMAGE="digits-trainer"
TAG="v1"
DISPLAY="mlops489-train-demo-${SUFFIX}"
AR_HOST="${REGION}-docker.pkg.dev"
IMAGE_URI="${AR_HOST}/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "Project:     ${PROJECT_ID}"
echo "Region:      ${REGION}"
echo "Repo:        ${REPO}  (will be deleted at the end)"
echo "Image:       ${IMAGE_URI}"
echo "Display:     ${DISPLAY}"
echo

JOB_ID=""
RENDERED_CONFIG=""

# Cleanup runs no matter what (success, error, or Ctrl-C).
cleanup() {
    echo
    echo "---- Cleanup ----"
    if [[ -n "${JOB_ID}" ]]; then
        echo "Cancelling Vertex AI job ${JOB_ID} if still running..."
        gcloud ai custom-jobs cancel "${JOB_ID}" --region="${REGION}" --quiet || true
    fi
    echo "Deleting Artifact Registry repo ${REPO}..."
    gcloud artifacts repositories delete "${REPO}" \
        --location="${REGION}" \
        --quiet || true
    if [[ -n "${RENDERED_CONFIG}" && -f "${RENDERED_CONFIG}" ]]; then
        rm -f "${RENDERED_CONFIG}"
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 2. Create a throwaway Artifact Registry repository
# ---------------------------------------------------------------------------
echo "---- Creating Artifact Registry repository ----"
gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="SE 489 training demo (auto-created, auto-deleted)"

# ---------------------------------------------------------------------------
# 3. Cloud Build: build the training image and push to AR
# ---------------------------------------------------------------------------
echo
echo "---- Submitting Cloud Build (slow step, ~3-5 min) ----"
gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPO=${REPO},_IMAGE=${IMAGE},_TAG=${TAG}"

# ---------------------------------------------------------------------------
# 4. Render config_cpu.yaml with the real project ID and demo repo, submit
# ---------------------------------------------------------------------------
echo
echo "---- Submitting Vertex AI custom job ----"
RENDERED_CONFIG=$(mktemp --suffix=.yaml)
sed -e "s|<project-id>|${PROJECT_ID}|g" \
    -e "s|mlops489-docker|${REPO}|g" \
    config_cpu.yaml > "${RENDERED_CONFIG}"

JOB_NAME=$(gcloud ai custom-jobs create \
    --region="${REGION}" \
    --display-name="${DISPLAY}" \
    --config="${RENDERED_CONFIG}" \
    --format="value(name)")

# `name` is a full resource path: projects/.../customJobs/<id>. Take the
# last path segment.
JOB_ID="${JOB_NAME##*/}"
echo "Submitted Vertex AI job: ${JOB_ID}"

# ---------------------------------------------------------------------------
# 5. Stream the job logs
# ---------------------------------------------------------------------------
echo
echo "---- Streaming logs (blocks until the job ends) ----"
gcloud ai custom-jobs stream-logs "${JOB_ID}" --region="${REGION}"

echo
echo "Done. (Cleanup runs next.)"
