# Exercises/GCPArtifactRegistry/demo.ps1 - Windows PowerShell end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes the
# digits-svc image via Cloud Build, lists the image, pulls it back locally,
# and ALWAYS deletes the repository at the end (even on failure).
#
# Run from inside Exercises/GCPArtifactRegistry/ with gcloud already
# authenticated and an active project set.
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
$Region   = 'us-central1'
$Suffix   = -join ((1..6) | ForEach-Object { [char[]]'abcdefghijklmnopqrstuvwxyz0123456789' | Get-Random })
$Repo     = "mlops489-ar-demo-$Suffix"
$Image    = 'digits-svc'
$Tag      = 'v1'
$ArHost   = "$Region-docker.pkg.dev"
$ImageUri = "$ArHost/$ProjectId/$Repo/$Image`:$Tag"

Write-Host "Project:   $ProjectId"
Write-Host "Region:    $Region"
Write-Host "Repo:      $Repo  (will be deleted at the end)"
Write-Host "Image:     $ImageUri"
Write-Host ''

function Invoke-Cleanup {
    Write-Host ''
    Write-Host "---- Cleaning up repository $Repo ----"
    try {
        gcloud artifacts repositories delete $Repo `
            --location=$Region `
            --quiet 2>$null
    } catch {
        # already gone, ignore
    }
}

try {
    # -----------------------------------------------------------------------
    # 2. Create the Artifact Registry repository
    # -----------------------------------------------------------------------
    Write-Host '---- Creating Artifact Registry repository ----'
    gcloud artifacts repositories create $Repo `
        --repository-format=docker `
        --location=$Region `
        --description='SE 489 demo (auto-created, auto-deleted)'
    if ($LASTEXITCODE -ne 0) { throw 'repository create failed' }

    # -----------------------------------------------------------------------
    # 3. Submit a Cloud Build that pushes the image to the new repo
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Submitting Cloud Build (this is the slow step) ----'
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions="_REGION=$Region,_REPO=$Repo,_IMAGE=$Image,_TAG=$Tag"
    if ($LASTEXITCODE -ne 0) { throw 'cloud build failed' }

    # -----------------------------------------------------------------------
    # 4. List the image in Artifact Registry
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Listing images in the repository ----'
    gcloud artifacts docker images list "$ArHost/$ProjectId/$Repo" --include-tags
    if ($LASTEXITCODE -ne 0) { throw 'image list failed' }

    # -----------------------------------------------------------------------
    # 5. Configure Docker for the regional Artifact Registry host and pull
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Configuring local Docker auth and pulling the image ----'
    gcloud auth configure-docker $ArHost --quiet
    if ($LASTEXITCODE -ne 0) { throw 'configure-docker failed' }
    docker pull $ImageUri
    if ($LASTEXITCODE -ne 0) { throw 'docker pull failed - is the Docker daemon running?' }

    Write-Host ''
    Write-Host 'Done. Local Docker image left on disk (docker rmi to remove).'
}
finally {
    Invoke-Cleanup
}
