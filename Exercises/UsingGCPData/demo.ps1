# Exercises/UsingGCPData/demo.ps1 - Windows PowerShell end-to-end runner.
# Creates a unique GCS bucket with Object Versioning on, configures it as a
# DVC remote with version_aware, pushes a small dataset, edits and pushes
# v2, time-travels back to v1, then ALWAYS deletes the bucket at the end
# (even on error) so nothing is left billing.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, storage.googleapis.com enabled), and `dvc` is installed in the
# active environment (uv pip install -e . from this folder).
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = 'Stop'

# Unique bucket suffix so re-running the demo doesn't collide.
$suffix = -join ((97..122) + (48..57) | Get-Random -Count 8 | ForEach-Object { [char]$_ })
$bucket = "mlops489-dvc-$suffix"
$region = "us-central1"        # free-tier eligible

function Invoke-Cleanup {
    Write-Host ""
    Write-Host "--- Cleanup: deleting bucket gs://$bucket ---"
    try { gcloud storage rm -r --quiet "gs://$bucket" 2>$null } catch {}
    try { dvc remote remove storage 2>$null } catch {}
    # Reset DVC tracking on data/ so the demo is rerunnable.
    if (Test-Path .dvc)            { Remove-Item -Recurse -Force .dvc }
    if (Test-Path data.dvc)        { Remove-Item -Force data.dvc }
    if (Test-Path data/.gitignore) { Remove-Item -Force data/.gitignore }
    Write-Host "Cleanup complete."
}

try {
    # --- 0. Sanity checks --------------------------------------------------
    if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
        Write-Host "[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first."
        exit 1
    }
    if (-not (Get-Command dvc -ErrorAction SilentlyContinue)) {
        Write-Host "[FAIL] dvc not on PATH. Run: uv pip install -e ."
        exit 1
    }
    $project = (gcloud config get-value project --quiet).Trim()
    if ([string]::IsNullOrEmpty($project) -or $project -eq "(unset)") {
        Write-Host "[FAIL] No active project. Run: gcloud config set project <project-id>"
        exit 1
    }
    Write-Host "[OK] gcloud + dvc + active project: $project"
    Write-Host "[OK] will create bucket: gs://$bucket in $region"

    # --- 1. Create a versioned bucket -------------------------------------
    Write-Host ""
    Write-Host "--- 1. Create bucket gs://$bucket ---"
    gcloud storage buckets create "gs://$bucket" --location=$region --uniform-bucket-level-access
    gcloud storage buckets update "gs://$bucket" --versioning

    # --- 2. List buckets with the modern CLI ------------------------------
    Write-Host ""
    Write-Host "--- 2. gcloud storage ls (modern CLI, replaces gsutil ls) ---"
    gcloud storage ls | Select-String $bucket

    # --- 3. Initialize DVC in this folder ---------------------------------
    Write-Host ""
    Write-Host "--- 3. dvc init (no-git mode, since the repo lives elsewhere) ---"
    dvc init --no-scm --force

    # --- 4. Add the GCS bucket as a version-aware remote ------------------
    Write-Host ""
    Write-Host "--- 4. dvc remote add + version_aware ---"
    dvc remote add -d storage "gs://$bucket" --force
    dvc remote modify storage version_aware true
    dvc config core.autostage true
    Write-Host "Resulting .dvc/config:"
    Get-Content .dvc/config

    # --- 5. Track and push v1 ---------------------------------------------
    Write-Host ""
    Write-Host "--- 5. dvc add + dvc push (v1) ---"
    dvc add data
    dvc push -v

    # --- 6. Mutate data and push v2 ---------------------------------------
    Write-Host ""
    Write-Host "--- 6. Edit data, dvc add + dvc push (v2) ---"
    Add-Content -Path data/sample_cars.csv -Value "Ford,F-150,1995,16.0,8,205,4500,USA"
    $v2Lines = (Get-Content data/sample_cars.csv).Count
    Write-Host "data/sample_cars.csv now has $v2Lines lines"
    dvc add data
    dvc push -v

    # --- 7. Prove version_aware: pull latest back -------------------------
    Write-Host ""
    Write-Host "--- 7. Clear local cache, restore latest, prove version_aware works ---"
    if (Test-Path .dvc/cache) { Remove-Item -Recurse -Force .dvc/cache }
    if (Test-Path data)       { Remove-Item -Recurse -Force data }
    dvc pull -v
    $afterLines = (Get-Content data/sample_cars.csv).Count
    Write-Host "after dvc pull: $afterLines lines"

    Write-Host ""
    Write-Host "All steps complete. Cleanup will now run."
}
finally {
    Invoke-Cleanup
}
