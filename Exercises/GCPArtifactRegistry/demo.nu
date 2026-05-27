#!/usr/bin/env nu
# Exercises/GCPArtifactRegistry/demo.nu - cross-platform end-to-end runner.
# Creates a uniquely-named Artifact Registry repository, builds and pushes the
# digits-svc image via Cloud Build, lists the image, pulls it back locally,
# and ALWAYS deletes the repository at the end (even on failure).
#
# Run from inside Exercises/GCPArtifactRegistry/ with gcloud already
# authenticated and an active project set.
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
let repo = $"mlops489-ar-demo-($suffix)"
let image = "digits-svc"
let tag = "v1"
let ar_host = $"($region)-docker.pkg.dev"
let image_uri = $"($ar_host)/($project_id)/($repo)/($image):($tag)"

print $"Project:   ($project_id)"
print $"Region:    ($region)"
print $"Repo:      ($repo)  (will be deleted at the end)"
print $"Image:     ($image_uri)"
print ""

# Always clean up. Defined up-front and called at the bottom whether the build
# succeeded or failed.
def cleanup [r: string, reg: string] {
    print ""
    print $"---- Cleaning up repository ($r) ----"
    do { gcloud artifacts repositories delete $r --location=$reg --quiet } | ignore
}

# ---------------------------------------------------------------------------
# 2. Create the Artifact Registry repository
# ---------------------------------------------------------------------------
print "---- Creating Artifact Registry repository ----"
try {
    gcloud artifacts repositories create $repo `
        --repository-format=docker `
        --location=$region `
        --description="SE 489 demo (auto-created, auto-deleted)"
} catch {
    print "Failed to create the repository. Aborting."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Submit a Cloud Build that pushes the image to the new repo
# ---------------------------------------------------------------------------
print ""
print "---- Submitting Cloud Build (this is the slow step) ----"
try {
    gcloud builds submit . `
        --config=cloudbuild.yaml `
        --substitutions=$"_REGION=($region),_REPO=($repo),_IMAGE=($image),_TAG=($tag)"
} catch {
    print "Cloud Build failed."
    cleanup $repo $region
    exit 1
}

# ---------------------------------------------------------------------------
# 4. List the image in Artifact Registry
# ---------------------------------------------------------------------------
print ""
print "---- Listing images in the repository ----"
try {
    gcloud artifacts docker images list $"($ar_host)/($project_id)/($repo)" --include-tags
} catch {
    print "Failed to list images."
    cleanup $repo $region
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Configure Docker for the regional Artifact Registry host and pull
# ---------------------------------------------------------------------------
print ""
print "---- Configuring local Docker auth and pulling the image ----"
try {
    gcloud auth configure-docker $ar_host --quiet
    docker pull $image_uri
} catch {
    print "Pull failed. Is the Docker daemon running?"
    cleanup $repo $region
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Clean up (always runs)
# ---------------------------------------------------------------------------
cleanup $repo $region
print ""
print "Done. Repository deleted; local Docker image left on disk (run `docker rmi` to remove)."
