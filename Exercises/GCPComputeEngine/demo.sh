#!/usr/bin/env bash
# Exercises/GCPComputeEngine/demo.sh - bash end-to-end runner.
# Creates two VMs in your active GCP project, lists them, runs a quick
# command over SSH, then ALWAYS stops + deletes them (even on error) so
# nothing is left burning credits.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, compute.googleapis.com enabled).
#
# Read-only checks first, then mutating commands.
set -uo pipefail

ZONE="us-central1-a"
VM_CPU="mlops489-cpu"
VM_PYTORCH="mlops489-pytorch"
PROJECT="$(gcloud config get-value project --quiet 2>/dev/null | tr -d '[:space:]')"

cleanup() {
    echo ""
    echo "--- Cleanup: stopping and deleting VMs (don't leave them billing) ---"
    gcloud compute instances delete "$VM_CPU"     --zone="$ZONE" --quiet 2>/dev/null || true
    gcloud compute instances delete "$VM_PYTORCH" --zone="$ZONE" --quiet 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT

# --- 0. Sanity checks ------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
    echo "[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first."
    exit 1
fi
if [ -z "$PROJECT" ] || [ "$PROJECT" = "(unset)" ]; then
    echo "[FAIL] No active project. Run: gcloud config set project <project-id>"
    exit 1
fi
echo "[OK] gcloud + active project: $PROJECT"

# --- 1. Create a free-tier e2-micro VM -------------------------------------
echo ""
echo "--- 1. Create e2-micro CPU VM ($VM_CPU in $ZONE) ---"
gcloud compute instances create "$VM_CPU" \
    --zone="$ZONE" \
    --machine-type=e2-micro \
    --image-family=debian-12 \
    --image-project=debian-cloud

# --- 2. List instances -----------------------------------------------------
echo ""
echo "--- 2. gcloud compute instances list ---"
gcloud compute instances list

# --- 3. SSH and run one command -------------------------------------------
echo ""
echo "--- 3. SSH into $VM_CPU and check Python ---"
# --strict-host-key-checking=no skips the y/n prompt on first connect.
gcloud compute ssh "$VM_CPU" --zone="$ZONE" --quiet \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --command="echo 'hostname:' \$(hostname); python3 --version || echo 'python3 not installed (expected on bare Debian)'"

# --- 4. Create a PyTorch Deep Learning VM (CPU image) ----------------------
echo ""
echo "--- 4. Create PyTorch Deep Learning VM ($VM_PYTORCH) ---"
# For GPU, add (and ensure you have GPU quota in this zone):
#   --accelerator="type=nvidia-tesla-t4,count=1" \
#   --maintenance-policy=TERMINATE \
#   --metadata="install-nvidia-driver=True"
gcloud compute instances create "$VM_PYTORCH" \
    --zone="$ZONE" \
    --image-family=pytorch-latest-cpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-4

# --- 5. List Deep Learning Containers (Artifact Registry replacement) ------
echo ""
echo "--- 5. List Deep Learning Containers in Artifact Registry ---"
# Old: gcloud container images list --repository=gcr.io/deeplearning-platform-release  (Container Registry, shut down 2025-03-18)
# New: Artifact Registry, same images at us-docker.pkg.dev
gcloud artifacts docker images list \
    us-docker.pkg.dev/deeplearning-platform-release/gcr.io \
    --include-tags --limit=10 || true

echo ""
echo "All steps complete. Cleanup will now run."
# trap will fire on exit
