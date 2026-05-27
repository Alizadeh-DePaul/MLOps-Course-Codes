#!/usr/bin/env nu
# Exercises/GCPComputeEngine/demo.nu - cross-platform end-to-end runner.
# Creates two VMs in your active GCP project, lists them, runs a quick
# command over SSH, then ALWAYS stops + deletes them at the end (even on
# error) so nothing is left burning credits.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, compute.googleapis.com enabled).

$env.config.error_style = "fancy"

let zone = "us-central1-a"
let vm_cpu = "mlops489-cpu"
let vm_pytorch = "mlops489-pytorch"

def cleanup [zone: string, vm_cpu: string, vm_pytorch: string] {
    print ""
    print "--- Cleanup: stopping and deleting VMs (don't leave them billing) ---"
    try { ^gcloud compute instances delete $vm_cpu     --zone=$zone --quiet }
    try { ^gcloud compute instances delete $vm_pytorch --zone=$zone --quiet }
    print "Cleanup complete."
}

# --- 0. Sanity checks ------------------------------------------------------
if (which gcloud | is-empty) {
    print "[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first."
    exit 1
}
let project = (^gcloud config get-value project --quiet | str trim)
if ($project | is-empty) or ($project == "(unset)") {
    print "[FAIL] No active project. Run: gcloud config set project <project-id>"
    exit 1
}
print $"[OK] gcloud + active project: ($project)"

# Wrap the whole flow in try so cleanup always runs.
try {
    # --- 1. Create a free-tier e2-micro VM ---------------------------------
    print ""
    print $"--- 1. Create e2-micro CPU VM \(($vm_cpu) in ($zone)\) ---"
    ^gcloud compute instances create $vm_cpu --zone=$zone --machine-type=e2-micro --image-family=debian-12 --image-project=debian-cloud

    # --- 2. List instances -------------------------------------------------
    print ""
    print "--- 2. gcloud compute instances list ---"
    ^gcloud compute instances list

    # --- 3. SSH and run one command ---------------------------------------
    print ""
    print $"--- 3. SSH into ($vm_cpu) and check Python ---"
    ^gcloud compute ssh $vm_cpu --zone=$zone --quiet --ssh-flag="-o StrictHostKeyChecking=no" --command="echo hostname: $(hostname); python3 --version || echo 'python3 not installed (expected on bare Debian)'"

    # --- 4. Create a PyTorch Deep Learning VM (CPU image) ------------------
    print ""
    print $"--- 4. Create PyTorch Deep Learning VM \(($vm_pytorch)\) ---"
    # For GPU, add (and ensure you have GPU quota in this zone):
    #   --accelerator="type=nvidia-tesla-t4,count=1"
    #   --maintenance-policy=TERMINATE
    #   --metadata="install-nvidia-driver=True"
    ^gcloud compute instances create $vm_pytorch --zone=$zone --image-family=pytorch-latest-cpu --image-project=deeplearning-platform-release --machine-type=n1-standard-4

    # --- 5. List Deep Learning Containers (Artifact Registry replacement) --
    print ""
    print "--- 5. List Deep Learning Containers in Artifact Registry ---"
    # Old: gcloud container images list --repository=gcr.io/deeplearning-platform-release  (Container Registry, shut down 2025-03-18)
    # New: Artifact Registry, same images at us-docker.pkg.dev
    try { ^gcloud artifacts docker images list us-docker.pkg.dev/deeplearning-platform-release/gcr.io --include-tags --limit=10 }

    print ""
    print "All steps complete. Cleanup will now run."
} catch {
    print ""
    print "A step failed. Running cleanup anyway."
}

cleanup $zone $vm_cpu $vm_pytorch
