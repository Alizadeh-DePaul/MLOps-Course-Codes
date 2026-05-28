#!/usr/bin/env nu
# Exercises/GCPTrainingModels/demo.nu - cross-platform end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes
# the training image via Cloud Build, submits a Vertex AI custom job,
# streams its logs, and ALWAYS cleans up at the end (cancels the job if
# still running, deletes the repository) - even if a step in the middle
# fails.
#
# Run from inside Exercises/GCPTrainingModels/ with gcloud already
# authenticated and an active project set. Requires the aiplatform,
# artifactregistry, and cloudbuild APIs to be enabled.
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
let repo = $"mlops489-train-demo-($suffix)"
let image = "digits-trainer"
let tag = "v1"
let display = $"mlops489-train-demo-($suffix)"
let ar_host = $"($region)-docker.pkg.dev"
let image_uri = $"($ar_host)/($project_id)/($repo)/($image):($tag)"

print $"Project:     ($project_id)"
print $"Region:      ($region)"
print $"Repo:        ($repo)  (will be deleted at the end)"
print $"Image:       ($image_uri)"
print $"Display:     ($display)"
print ""

mut job_id = ""

# Cleanup runs no matter what.
def cleanup [r: string, reg: string, jid: string] {
    print ""
    print "---- Cleanup ----"
    if ($jid | str length) > 0 {
        print $"Cancelling Vertex AI job ($jid) if still running..."
        do { ^gcloud ai custom-jobs cancel $jid --region=$reg --quiet } | ignore
    }
    print $"Deleting Artifact Registry repo ($r)..."
    do { ^gcloud artifacts repositories delete $r --location=$reg --quiet } | ignore
}

# ---------------------------------------------------------------------------
# 2. Create a throwaway Artifact Registry repository
# ---------------------------------------------------------------------------
print "---- Creating Artifact Registry repository ----"
try {
    gcloud artifacts repositories create $repo `
        --repository-format=docker `
        --location=$region `
        --description="SE 489 training demo (auto-created, auto-deleted)"
} catch {
    print "Failed to create the repository. Aborting."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Cloud Build: build the training image and push to AR
# ---------------------------------------------------------------------------
print ""
print "---- Submitting Cloud Build (slow step, ~3-5 min) ----"
try {
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions=$"_REGION=($region),_REPO=($repo),_IMAGE=($image),_TAG=($tag)"
} catch {
    print "Cloud Build failed."
    cleanup $repo $region $job_id
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Render config_cpu.yaml with the real project ID, image URI, and submit
# ---------------------------------------------------------------------------
print ""
print "---- Submitting Vertex AI custom job ----"
let rendered_config = (mktemp --suffix .yaml | str trim)
open config_cpu.yaml
    | str replace --all "<project-id>" $project_id
    | str replace --all "mlops489-docker" $repo
    | save --force $rendered_config

try {
    let create_output = (
        gcloud ai custom-jobs create
            --region=$region
            --display-name=$display
            --config=$rendered_config
            --format="value(name)"
    )
    # `name` is a full resource path: projects/.../customJobs/<id>. Grab the
    # last segment.
    $job_id = ($create_output | str trim | split row "/" | last)
    print $"Submitted Vertex AI job: ($job_id)"
} catch {
    print "Job submission failed."
    cleanup $repo $region $job_id
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Stream the job logs
# ---------------------------------------------------------------------------
print ""
print "---- Streaming logs (blocks until the job ends) ----"
try {
    gcloud ai custom-jobs stream-logs $job_id --region=$region
} catch {
    print "stream-logs failed or job exited non-zero."
    cleanup $repo $region $job_id
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Cleanup (always runs)
# ---------------------------------------------------------------------------
cleanup $repo $region $job_id
print ""
print "Done."
