# Exercises/DeploymentGCPCloudFunctions/demo.ps1 - Windows PowerShell end-to-end runner.
# Trains a KNN model, uploads it to a throwaway GCS bucket, deploys two HTTP
# Cloud Run functions (hello + knn), calls both, and ALWAYS tears everything
# down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudFunctions/ with gcloud already
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
$Region    = 'us-central1'
$Suffix    = -join ((1..6) | ForEach-Object { [char[]]'abcdefghijklmnopqrstuvwxyz0123456789' | Get-Random })
$Bucket    = "mlops489-cf-demo-$Suffix"
$ModelFile = 'model.pkl'
$HelloFn   = "hello-mlops-$Suffix"
$KnnFn     = "knn-classifier-$Suffix"
$Runtime   = 'python311'

Write-Host "Project:   $ProjectId"
Write-Host "Region:    $Region"
Write-Host "Bucket:    gs://$Bucket  (will be deleted at the end)"
Write-Host "Functions: $HelloFn, $KnnFn  (will be deleted at the end)"
Write-Host ''

function Invoke-Cleanup {
    Write-Host ''
    Write-Host '---- Cleaning up ----'
    try { gcloud functions delete $HelloFn --region=$Region --gen2 --quiet 2>$null } catch {}
    try { gcloud functions delete $KnnFn   --region=$Region --gen2 --quiet 2>$null } catch {}
    try { gcloud storage rm --recursive "gs://$Bucket" --quiet 2>$null } catch {}
}

try {
    # -----------------------------------------------------------------------
    # 2. Environment + train the model
    # -----------------------------------------------------------------------
    Write-Host '---- Setting up the environment and training the model ----'
    uv venv                                 # alt: python -m venv .venv
    . .\.venv\Scripts\Activate.ps1
    uv pip install -e .                      # alt: pip install -e .
    python train_model.py                    # writes model.pkl

    # -----------------------------------------------------------------------
    # 3. Create a bucket and upload the model
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Creating bucket and uploading the model ----'
    gcloud storage buckets create "gs://$Bucket" --location=$Region
    if ($LASTEXITCODE -ne 0) { throw 'bucket create failed' }
    gcloud storage cp $ModelFile "gs://$Bucket/$ModelFile"
    if ($LASTEXITCODE -ne 0) { throw 'model upload failed' }

    # -----------------------------------------------------------------------
    # 4. Deploy + call the hello function
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Deploying hello function (this is slow) ----'
    gcloud functions deploy $HelloFn `
        --gen2 --runtime=$Runtime --region=$Region `
        --source=hello --entry-point=hello_mlops `
        --trigger-http --allow-unauthenticated
    if ($LASTEXITCODE -ne 0) { throw 'hello deploy failed' }
    $HelloUrl = (gcloud functions describe $HelloFn --region=$Region --gen2 `
        --format="value(serviceConfig.uri)").Trim()
    Write-Host "hello URL: $HelloUrl"
    curl.exe -fsS "$HelloUrl`?name=MLOPS%20engineer"; Write-Host ''

    # -----------------------------------------------------------------------
    # 5. Deploy + call the knn function
    # -----------------------------------------------------------------------
    Write-Host ''
    Write-Host '---- Deploying knn function (this is slow) ----'
    gcloud functions deploy $KnnFn `
        --gen2 --runtime=$Runtime --region=$Region `
        --source=knn --entry-point=knn_classifier `
        --trigger-http --allow-unauthenticated `
        --set-env-vars="BUCKET_NAME=$Bucket,MODEL_FILE=$ModelFile"
    if ($LASTEXITCODE -ne 0) { throw 'knn deploy failed' }
    $KnnUrl = (gcloud functions describe $KnnFn --region=$Region --gen2 `
        --format="value(serviceConfig.uri)").Trim()
    Write-Host "knn URL: $KnnUrl"
    curl.exe -fsS -X POST "$KnnUrl" `
        -H "Content-Type: application/json" `
        -d '{"input_data": "5.1,3.5,1.4,0.2"}'; Write-Host ''

    Write-Host ''
    Write-Host 'Done. All cloud resources will be deleted on exit; model.pkl is left on disk.'
}
finally {
    Invoke-Cleanup
}
