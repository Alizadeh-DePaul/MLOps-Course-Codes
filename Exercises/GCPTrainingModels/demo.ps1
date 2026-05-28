# Exercises/GCPTrainingModels/demo.ps1 - Windows PowerShell end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes
# the training image via Cloud Build, submits an Agent Platform custom job,
# streams its logs, and ALWAYS cleans up at the end (cancels the job if
# still running, deletes the repository) - even if a step in the middle
# fails.
#
# Run from inside Exercises/GCPTrainingModels/ with gcloud already
# authenticated and an active project set. Requires the aiplatform,
# artifactregistry, and cloudbuild APIs to be enabled.
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
$Repo     = "mlops489-train-demo-$Suffix"
$Image    = 'digits-trainer'
$Tag      = 'v1'
$Display  = "mlops489-train-demo-$Suffix"
$ArHost   = "$Region-docker.pkg.dev"
$ImageUri = "$ArHost/$ProjectId/$Repo/$Image`:$Tag"

Write-Host "Project:     $ProjectId"
Write-Host "Region:      $Region"
Write-Host "Repo:        $Repo  (will be deleted at the end)"
Write-Host "Image:       $ImageUri"
Write-Host "Display:     $Display"
Write-Host ''

$Script:JobId = ''
$Script:RenderedConfig = ''

function Invoke-Cleanup {
    Write-Host ''
    Write-Host '---- Cleanup ----'
    if (-not [string]::IsNullOrWhiteSpace($Script:JobId)) {
        Write-Host "Cancelling Agent Platform job $($Script:JobId) if still running..."
        try {
            gcloud ai custom-jobs cancel $Script:JobId --region=$Region --quiet 2>$null
        } catch {
            # already done, ignore
        }
    }
    Write-Host "Deleting Artifact Registry repo $Repo..."
    try {
        gcloud artifacts repositories delete $Repo `
            --location=$Region `
            --quiet 2>$null
    } catch {
        # already gone, ignore
    }
    if ((-not [string]::IsNullOrWhiteSpace($Script:RenderedConfig)) -and (Test-Path $Script:RenderedConfig)) {
        Remove-Item -Force $Script:RenderedConfig
    }
}

try {
    # -----------------------------------------------------------------------
    # 2. Create a throwaway Artifact Registry repository
    # -----------------------------------------------------------------------
    Write-Host '---- Creating Artifact Registry repository ----'
    gcloud artifacts repositories create $Repo `
        --repository-format=docker `
        --location=$Region `
        --description='SE 489 training demo (auto-created, auto-deleted)'
    if ($LASTEXITCODE -ne 0) { throw 'repository create failed' }

    # -----------------------------------------------------------------------
    # 3. Cloud Build: build the training image and push to AR
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Submitting Cloud Build (slow step, ~3-5 min) ----'
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions="_REGION=$Region,_REPO=$Repo,_IMAGE=$Image,_TAG=$Tag"
    if ($LASTEXITCODE -ne 0) { throw 'cloud build failed' }

    # -----------------------------------------------------------------------
    # 4. Render config_cpu.yaml with the real project ID and demo repo
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Submitting Agent Platform custom job ----'
    $Script:RenderedConfig = [System.IO.Path]::GetTempFileName() + '.yaml'
    (Get-Content config_cpu.yaml) `
        -replace '<project-id>', $ProjectId `
        -replace 'mlops489-docker', $Repo `
        | Set-Content $Script:RenderedConfig

    $JobName = gcloud ai custom-jobs create `
        --region=$Region `
        --display-name=$Display `
        --config=$Script:RenderedConfig `
        --format='value(name)'
    if ($LASTEXITCODE -ne 0) { throw 'custom-jobs create failed' }

    # name is a full resource path: projects/.../customJobs/<id>. Grab the
    # last segment.
    $Script:JobId = ($JobName.Trim() -split '/')[-1]
    Write-Host "Submitted Agent Platform job: $($Script:JobId)"

    # -----------------------------------------------------------------------
    # 5. Stream the job logs
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Streaming logs (blocks until the job ends) ----'
    gcloud ai custom-jobs stream-logs $Script:JobId --region=$Region
    if ($LASTEXITCODE -ne 0) { throw 'stream-logs failed or job exited non-zero' }

    Write-Host ''
    Write-Host 'Done. (Cleanup runs next.)'
}
finally {
    Invoke-Cleanup
}
