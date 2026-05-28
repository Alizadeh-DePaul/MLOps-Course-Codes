#!/usr/bin/env nu
# Exercises/DeploymentGCPCloudFunctions/demo.nu - cross-platform end-to-end runner.
# Trains a KNN model, uploads it to a throwaway GCS bucket, deploys two HTTP
# Cloud Run functions (hello + knn), calls both, and ALWAYS tears everything
# down at the end (even on failure).
#
# Run from inside Exercises/DeploymentGCPCloudFunctions/ with gcloud already
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
let bucket = $"mlops489-cf-demo-($suffix)"
let model_file = "model.pkl"
let hello_fn = $"hello-mlops-($suffix)"
let knn_fn = $"knn-classifier-($suffix)"
let runtime = "python311"

print $"Project:   ($project_id)"
print $"Region:    ($region)"
print $"Bucket:    gs://($bucket)  (will be deleted at the end)"
print $"Functions: ($hello_fn), ($knn_fn)  (will be deleted at the end)"
print ""

# Always clean up. Defined up-front and called whether the run succeeded or not.
def cleanup [hf: string, kf: string, b: string, reg: string] {
    print ""
    print "---- Cleaning up ----"
    do { gcloud functions delete $hf --region=$reg --gen2 --quiet } | ignore
    do { gcloud functions delete $kf --region=$reg --gen2 --quiet } | ignore
    do { gcloud storage rm --recursive $"gs://($b)" --quiet } | ignore
}

# ---------------------------------------------------------------------------
# 2. Environment + train the model
# ---------------------------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh  (macOS/Linux)
#                  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  (Windows)
print "---- Setting up the environment and training the model ----"
uv venv                                     # alt: python -m venv .venv

# Nushell does not source activation scripts; prepend the venv bin dir to PATH
# and set VIRTUAL_ENV ourselves. Works identically on Windows / macOS / Linux.
let venv_bin = if $nu.os-info.name == "windows" {
    (pwd | path join ".venv" "Scripts")
} else {
    (pwd | path join ".venv" "bin")
}
$env.PATH = ($env.PATH | prepend $venv_bin)
$env.VIRTUAL_ENV = (pwd | path join ".venv")

uv pip install -e .                         # alt: pip install -e .
python train_model.py                        # writes model.pkl

# ---------------------------------------------------------------------------
# 3. Create a bucket and upload the model
# ---------------------------------------------------------------------------
print ""
print "---- Creating bucket and uploading the model ----"
try {
    gcloud storage buckets create $"gs://($bucket)" --location=$region
    gcloud storage cp $model_file $"gs://($bucket)/($model_file)"
} catch {
    print "Bucket creation / upload failed."
    cleanup $hello_fn $knn_fn $bucket $region
    exit 1
}

# ---------------------------------------------------------------------------
# 4. Deploy + call the hello function
# ---------------------------------------------------------------------------
print ""
print "---- Deploying hello function (this is slow) ----"
try {
    gcloud functions deploy $hello_fn `
        --gen2 --runtime=$runtime --region=$region `
        --source=hello --entry-point=hello_mlops `
        --trigger-http --allow-unauthenticated
    let hello_url = (gcloud functions describe $hello_fn --region=$region --gen2 --format="value(serviceConfig.uri)" | str trim)
    print $"hello URL: ($hello_url)"
    curl -fsS $"($hello_url)?name=MLOPS%20engineer"
    print ""
} catch {
    print "Hello deploy/call failed."
    cleanup $hello_fn $knn_fn $bucket $region
    exit 1
}

# ---------------------------------------------------------------------------
# 5. Deploy + call the knn function
# ---------------------------------------------------------------------------
print ""
print "---- Deploying knn function (this is slow) ----"
try {
    gcloud functions deploy $knn_fn `
        --gen2 --runtime=$runtime --region=$region `
        --source=knn --entry-point=knn_classifier `
        --trigger-http --allow-unauthenticated `
        --set-env-vars=$"BUCKET_NAME=($bucket),MODEL_FILE=($model_file)"
    let knn_url = (gcloud functions describe $knn_fn --region=$region --gen2 --format="value(serviceConfig.uri)" | str trim)
    print $"knn URL: ($knn_url)"
    curl -fsS -X POST $knn_url -H "Content-Type: application/json" -d '{"input_data": "5.1,3.5,1.4,0.2"}'
    print ""
} catch {
    print "KNN deploy/call failed."
    cleanup $hello_fn $knn_fn $bucket $region
    exit 1
}

# ---------------------------------------------------------------------------
# 6. Clean up (always runs)
# ---------------------------------------------------------------------------
cleanup $hello_fn $knn_fn $bucket $region
print ""
print "Done. All cloud resources deleted; model.pkl left on disk."
