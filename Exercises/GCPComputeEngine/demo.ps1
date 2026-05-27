# Exercises/GCPComputeEngine/demo.ps1 - Windows PowerShell end-to-end runner.
# Creates two VMs in your active GCP project, lists them, runs a quick
# command over SSH, then ALWAYS stops + deletes them at the end (even on
# error) so nothing is left burning credits.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, compute.googleapis.com enabled).
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

$Zone      = 'us-central1-a'
$VmCpu     = 'mlops489-cpu'
$VmPytorch = 'mlops489-pytorch'

function Invoke-Cleanup {
    Write-Host ''
    Write-Host '--- Cleanup: stopping and deleting VMs (do not leave them billing) ---'
    & gcloud compute instances delete $VmCpu     --zone=$Zone --quiet 2>$null
    & gcloud compute instances delete $VmPytorch --zone=$Zone --quiet 2>$null
    Write-Host 'Cleanup complete.'
}

try {
    # --- 0. Sanity checks --------------------------------------------------
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Write-Host '[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first.'
        exit 1
    }
    $Project = (& gcloud config get-value project --quiet 2>$null).Trim()
    if ([string]::IsNullOrEmpty($Project) -or ($Project -eq '(unset)')) {
        Write-Host '[FAIL] No active project. Run: gcloud config set project <project-id>'
        exit 1
    }
    Write-Host "[OK] gcloud + active project: $Project"

    # --- 1. Create a free-tier e2-micro VM ---------------------------------
    Write-Host ''
    Write-Host "--- 1. Create e2-micro CPU VM ($VmCpu in $Zone) ---"
    & gcloud compute instances create $VmCpu `
        --zone=$Zone `
        --machine-type=e2-micro `
        --image-family=debian-12 `
        --image-project=debian-cloud

    # --- 2. List instances -------------------------------------------------
    Write-Host ''
    Write-Host '--- 2. gcloud compute instances list ---'
    & gcloud compute instances list

    # --- 3. SSH and run one command ---------------------------------------
    Write-Host ''
    Write-Host "--- 3. SSH into $VmCpu and check Python ---"
    & gcloud compute ssh $VmCpu --zone=$Zone --quiet `
        --ssh-flag='-o StrictHostKeyChecking=no' `
        --command="echo hostname: `$(hostname); python3 --version || echo 'python3 not installed (expected on bare Debian)'"

    # --- 4. Create a PyTorch Deep Learning VM (CPU image) ------------------
    Write-Host ''
    Write-Host "--- 4. Create PyTorch Deep Learning VM ($VmPytorch) ---"
    # For GPU, add (and ensure you have GPU quota in this zone):
    #   --accelerator="type=nvidia-tesla-t4,count=1"
    #   --maintenance-policy=TERMINATE
    #   --metadata="install-nvidia-driver=True"
    & gcloud compute instances create $VmPytorch `
        --zone=$Zone `
        --image-family=pytorch-latest-cpu `
        --image-project=deeplearning-platform-release `
        --machine-type=n1-standard-4

    # --- 5. List Deep Learning Containers (Artifact Registry replacement) --
    Write-Host ''
    Write-Host '--- 5. List Deep Learning Containers in Artifact Registry ---'
    # Old: gcloud container images list --repository=gcr.io/deeplearning-platform-release  (Container Registry, shut down 2025-03-18)
    # New: Artifact Registry, same images at us-docker.pkg.dev
    try {
        & gcloud artifacts docker images list `
            us-docker.pkg.dev/deeplearning-platform-release/gcr.io `
            --include-tags --limit=10
    } catch {
        Write-Host '(non-fatal: artifact registry listing failed; continuing)'
    }

    Write-Host ''
    Write-Host 'All steps complete. Cleanup will now run.'
}
finally {
    Invoke-Cleanup
}
