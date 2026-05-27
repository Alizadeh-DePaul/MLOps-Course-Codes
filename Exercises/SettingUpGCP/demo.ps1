# Exercises/SettingUpGCP/demo.ps1 - Windows PowerShell smoke test for the GCP setup.
# Run from inside Exercises/SettingUpGCP/. Read-only; does not modify your
# project or auth state. Checks:
#   1. gcloud is on PATH and runs
#   2. There is an active authenticated user
#   3. ADC (application-default credentials) work
#   4. An active project is configured
#   5. The Week 9 / 10 APIs are enabled on that project
# If Windows blocks execution, run once per terminal:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = 'Continue'

$requiredApis = @(
    'compute.googleapis.com',
    'storage.googleapis.com',
    'artifactregistry.googleapis.com',
    'cloudbuild.googleapis.com',
    'run.googleapis.com',
    'cloudfunctions.googleapis.com',
    'iam.googleapis.com',
    'aiplatform.googleapis.com'
)

$pass = 0
$fail = 0

function Check {
    param([string]$Label, [bool]$Ok, [string]$Hint)
    if ($Ok) {
        Write-Host "  [PASS] $Label"
        $script:pass++
    } else {
        Write-Host "  [FAIL] $Label"
        Write-Host "         fix: $Hint"
        $script:fail++
    }
}

Write-Host "GCP setup smoke test"
Write-Host "===================="

# --- 1. gcloud on PATH -----------------------------------------------------
$gcloudCmd = Get-Command gcloud -ErrorAction SilentlyContinue
if ($gcloudCmd) {
    Check "gcloud CLI is installed and on PATH" $true ""
} else {
    Check "gcloud CLI is installed and on PATH" $false "install from https://cloud.google.com/sdk/docs/install, then restart your terminal"
    Write-Host "Cannot continue without gcloud. Stopping."
    exit 1
}

# --- 2. Active authenticated user -----------------------------------------
$activeAccount = (gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>$null | Out-String).Trim()
if ($activeAccount) {
    Check "authenticated as: $activeAccount" $true ""
} else {
    Check "authenticated as: (none)" $false "run: gcloud auth login"
}

# --- 3. Application Default Credentials -----------------------------------
gcloud auth application-default print-access-token 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Check "application-default credentials work" $true ""
} else {
    Check "application-default credentials work" $false "run: gcloud auth application-default login"
}

# --- 4. Active project ----------------------------------------------------
$activeProject = (gcloud config get-value project --quiet 2>$null | Out-String).Trim()
$hasProject = $activeProject -and $activeProject -ne '(unset)'
if ($hasProject) {
    Check "active project: $activeProject" $true ""
} else {
    Check "active project: (unset)" $false "run: gcloud config set project <your-project-id>"
}

# --- 5. Required APIs enabled ---------------------------------------------
if ($hasProject) {
    $enabled = (gcloud services list --enabled --format='value(config.name)' 2>$null) -split "`r?`n"
    foreach ($api in $requiredApis) {
        if ($enabled -contains $api) {
            Check "API enabled: $api" $true ""
        } else {
            Check "API enabled: $api" $false "run: gcloud services enable $api"
        }
    }
} else {
    Write-Host "  (skipping API checks - no active project)"
}

Write-Host ""
Write-Host "Summary: $pass passed, $fail failed"
if ($fail -gt 0) {
    Write-Host "Re-run this script after fixing the failures above."
    exit 1
} else {
    Write-Host "All checks passed. Your GCP setup is ready."
}
