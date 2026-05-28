#!/usr/bin/env nu
# Exercises/DeploymentGCPCloudRun/demo.nu - cross-platform end-to-end runner.
# Creates a throwaway Artifact Registry repo, then builds + pushes + deploys the
# FastAPI container to Cloud Run via Cloud Build, calls the service, and ALWAYS
# tears everything down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudRun/ with gcloud already
# authenticated and an active project set. No local Docker needed.
$env.config.error_style = "fancy"

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
let project_id = (gcloud config get-value project | str trim)
if ($project_id | is-empty) {
    print "ERROR: No active project. Run `gcloud config set project <id>` first."
    exit 1
}
let region = "us-central1"
let suffix = (random chars --length 6 | str downcase)
let repo = $"mlops489-cr-demo-($suffix)"
let service = $"cloud-run-mlops-($suffix)"
let tag = "v1"

print $"Project:  ($project_id)"
print $"Region:   ($region)"
print $"Repo:     ($repo)  (will be deleted at the end)"
print $"Service:  ($service)  (will be deleted at the end)"
print ""

def cleanup [svc: string, r: string, reg: string] {
    print ""
    print "---- Cleaning up ----"
    do { gcloud run services delete $svc --region=$reg --quiet } | ignore
    do { gcloud artifacts repositories delete $r --location=$reg --quiet } | ignore
}

# ---------------------------------------------------------------------------
# 2. Create a throwaway Artifact Registry repository
# ---------------------------------------------------------------------------
print "---- Creating Artifact Registry repository ----"
try {
    gcloud artifacts repositories create $repo `
        --repository-format=docker `
        --location=$region `
        --description="SE 489 Cloud Run demo (auto-created, auto-deleted)"
} catch {
    print "Failed to create the repository. Aborting."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Build + push + deploy in one Cloud Build run
# ---------------------------------------------------------------------------
print ""
print "---- Build + push + deploy via Cloud Build (this is the slow step) ----"
try {
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions=$"_REGION=($region),_REPO=($repo),_SERVICE=($service),_TAG=($tag)"
} catch {
    print "Cloud Build failed."
    cleanup $service $repo $region
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Call the deployed service
# ---------------------------------------------------------------------------
print ""
print "---- Calling the service ----"
try {
    let service_url = (gcloud run services describe $service --region=$region --format="value(status.url)" | str trim)
    print $"Service URL: ($service_url)"
    curl -fsS $"($service_url)/"
    print ""
    curl -fsS $"($service_url)/items/1"
    print ""
} catch {
    print "Service call failed."
    cleanup $service $repo $region
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Clean up (always runs)
# ---------------------------------------------------------------------------
cleanup $service $repo $region
print ""
print "Done. All cloud resources deleted."
