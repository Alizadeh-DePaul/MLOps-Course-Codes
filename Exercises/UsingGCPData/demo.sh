#!/usr/bin/env bash
# Exercises/UsingGCPData/demo.sh - bash end-to-end runner.
# Creates a unique GCS bucket with Object Versioning on, configures it as a
# DVC remote with version_aware, pushes a small dataset, edits and pushes
# v2, time-travels back to v1, then ALWAYS deletes the bucket at the end
# (even on error) so nothing is left billing.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, storage.googleapis.com enabled), and `dvc` is installed in the
# active environment (uv pip install -e . from this folder).

set -euo pipefail

# Unique bucket suffix so re-running the demo doesn't collide.
SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' </dev/urandom | head -c 8)
BUCKET="mlops489-dvc-${SUFFIX}"
REGION="us-central1"        # free-tier eligible

cleanup() {
    echo ""
    echo "--- Cleanup: deleting bucket gs://${BUCKET} ---"
    gcloud storage rm -r --quiet "gs://${BUCKET}" 2>/dev/null || true
    # Best-effort: also drop the DVC remote so we don't leave stale config
    dvc remote remove storage >/dev/null 2>&1 || true
    # And reset DVC tracking on data/ so the demo is rerunnable.
    rm -rf .dvc data.dvc data/.gitignore
    echo "Cleanup complete."
}
trap cleanup EXIT

# --- 0. Sanity checks ------------------------------------------------------
command -v gcloud >/dev/null 2>&1 || { echo "[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first."; exit 1; }
command -v dvc    >/dev/null 2>&1 || { echo "[FAIL] dvc not on PATH. Run: uv pip install -e ."; exit 1; }

PROJECT=$(gcloud config get-value project --quiet | tr -d '[:space:]')
if [ -z "${PROJECT}" ] || [ "${PROJECT}" = "(unset)" ]; then
    echo "[FAIL] No active project. Run: gcloud config set project <project-id>"
    exit 1
fi
echo "[OK] gcloud + dvc + active project: ${PROJECT}"
echo "[OK] will create bucket: gs://${BUCKET} in ${REGION}"

# --- 1. Create a versioned bucket -----------------------------------------
echo ""
echo "--- 1. Create bucket gs://${BUCKET} ---"
gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --uniform-bucket-level-access
gcloud storage buckets update "gs://${BUCKET}" --versioning

# --- 2. List buckets with the modern CLI ----------------------------------
echo ""
echo "--- 2. gcloud storage ls (modern CLI, replaces gsutil ls) ---"
gcloud storage ls | grep "${BUCKET}" || true

# --- 3. Initialize DVC in this folder -------------------------------------
echo ""
echo "--- 3. dvc init (no-git mode, since the repo lives elsewhere) ---"
dvc init --no-scm --force

# --- 4. Add the GCS bucket as a version-aware remote ----------------------
echo ""
echo "--- 4. dvc remote add + version_aware ---"
dvc remote add -d storage "gs://${BUCKET}" --force
dvc remote modify storage version_aware true
dvc config core.autostage true
echo "Resulting .dvc/config:"
cat .dvc/config

# --- 5. Track and push v1 -------------------------------------------------
echo ""
echo "--- 5. dvc add + dvc push (v1) ---"
dvc add data
dvc push -v

# --- 6. Mutate data and push v2 -------------------------------------------
echo ""
echo "--- 6. Edit data, dvc add + dvc push (v2) ---"
printf "Ford,F-150,1995,16.0,8,205,4500,USA\n" >> data/sample_cars.csv
V2_LINES=$(wc -l < data/sample_cars.csv)
echo "data/sample_cars.csv now has ${V2_LINES} lines"
dvc add data
dvc push -v

# --- 7. Prove version_aware: pull v1 back ---------------------------------
echo ""
echo "--- 7. Clear local cache, restore latest, prove version_aware works ---"
rm -rf .dvc/cache data
dvc pull -v
AFTER_LINES=$(wc -l < data/sample_cars.csv)
echo "after dvc pull: ${AFTER_LINES} lines"

echo ""
echo "All steps complete. Cleanup will now run."
