# Exercises/DeploymentGCPCloudRun/demo.ps1 - Windows PowerShell end-to-end runner.
# Creates a throwaway Artifact Registry repo, then builds + pushes + deploys the
# FastAPI container to Cloud Run via Cloud Build, calls the service, and ALWAYS
# tears everything down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudRun/ with gcloud already
# authenticated and an active project set. No local Docker needed.
#
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
$ProjectId = (gcloud config get-value project 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    Write-Host "ERROR: No active project. Run 'gcloud config set project <id>' first."
    exit 1
}
$Region  = 'us-central1'
$Suffix  = -join ((1..6) | ForEach-Object { [char[]]'abcdefghijklmnopqrstuvwxyz0123456789' | Get-Random })
$Repo    = "mlops489-cr-demo-$Suffix"
$Service = "cloud-run-mlops-$Suffix"
$Tag     = 'v1'

Write-Host "Project:  $ProjectId"
Write-Host "Region:   $Region"
Write-Host "Repo:     $Repo  (will be deleted at the end)"
Write-Host "Service:  $Service  (will be deleted at the end)"
Write-Host ''

function Invoke-Cleanup {
    Write-Host ''
    Write-Host '---- Cleaning up ----'
    try { gcloud run services delete $Service --region=$Region --quiet 2>$null } catch {}
    try { gcloud artifacts repositories delete $Repo --location=$Region --quiet 2>$null } catch {}
}

try {
    # -----------------------------------------------------------------------
    # 2. Create a throwaway Artifact Registry repository
    # -----------------------------------------------------------------------
    Write-Host '---- Creating Artifact Registry repository ----'
    gcloud artifacts repositories create $Repo `
        --repository-format=docker `
        --location=$Region `
        --description='SE 489 Cloud Run demo (auto-created, auto-deleted)'
    if ($LASTEXITCODE -ne 0) { throw 'repository create failed' }

    # -----------------------------------------------------------------------
    # 3. Build + push + deploy in one Cloud Build run
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Build + push + deploy via Cloud Build (this is the slow step) ----'
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions="_REGION=$Region,_REPO=$Repo,_SERVICE=$Service,_TAG=$Tag"
    if ($LASTEXITCODE -ne 0) { throw 'cloud build failed' }

    # -----------------------------------------------------------------------
    # 4. Call the deployed service
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Calling the service ----'
    $ServiceUrl = (gcloud run services describe $Service --region=$Region `
        --format="value(status.url)").Trim()
    Write-Host "Service URL: $ServiceUrl"
    curl.exe -fsS "$ServiceUrl/";        Write-Host ''
    curl.exe -fsS "$ServiceUrl/items/1"; Write-Host ''

    Write-Host ''
    Write-Host 'Done. All cloud resources will be deleted on exit.'
}
finally {
    Invoke-Cleanup
}
