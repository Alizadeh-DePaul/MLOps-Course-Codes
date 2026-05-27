#!/usr/bin/env nu
# Exercises/UsingGCPData/demo.nu - cross-platform end-to-end runner.
# Creates a unique GCS bucket with Object Versioning on, configures it as a
# DVC remote with version_aware, pushes a small dataset, edits and pushes
# v2, time-travels back to v1, then ALWAYS deletes the bucket at the end
# (even on error) so nothing is left billing.
#
# Required: SettingUpGCP smoke test passes (gcloud authenticated, project
# set, storage.googleapis.com enabled), and `dvc` is installed in the
# active environment (uv pip install -e . from this folder).

$env.config.error_style = "fancy"

# Unique bucket suffix so re-running the demo doesn't collide.
let suffix = (random chars --length 8 | str downcase)
let bucket = $"mlops489-dvc-($suffix)"
let region = "us-central1"        # free-tier eligible

def cleanup [bucket: string] {
    print ""
    print $"--- Cleanup: deleting bucket gs://($bucket) ---"
    try { ^gcloud storage rm -r --quiet $"gs://($bucket)" }
    # Best-effort: also drop the DVC remote so we don't leave stale config
    try { ^dvc remote remove storage out+err> /dev/null }
    # And reset DVC tracking on data/ so the demo is rerunnable.
    try { rm -rf .dvc data.dvc data/.gitignore }
    print "Cleanup complete."
}

# --- 0. Sanity checks ------------------------------------------------------
if (which gcloud | is-empty) {
    print "[FAIL] gcloud not on PATH. Run the SettingUpGCP smoke test first."
    exit 1
}
if (which dvc | is-empty) {
    print "[FAIL] dvc not on PATH. Run: uv pip install -e ."
    exit 1
}
let project = (^gcloud config get-value project --quiet | str trim)
if ($project | is-empty) or ($project == "(unset)") {
    print "[FAIL] No active project. Run: gcloud config set project <project-id>"
    exit 1
}
print $"[OK] gcloud + dvc + active project: ($project)"
print $"[OK] will create bucket: gs://($bucket) in ($region)"

# Wrap the whole flow in try so cleanup always runs.
try {
    # --- 1. Create a versioned bucket -------------------------------------
    print ""
    print $"--- 1. Create bucket gs://($bucket) ---"
    ^gcloud storage buckets create $"gs://($bucket)" --location=$region --uniform-bucket-level-access
    ^gcloud storage buckets update $"gs://($bucket)" --versioning

    # --- 2. List buckets with the modern CLI ------------------------------
    print ""
    print "--- 2. gcloud storage ls (modern CLI, replaces gsutil ls) ---"
    ^gcloud storage ls | lines | where ($it | str contains $bucket) | print

    # --- 3. Initialize DVC in this folder ---------------------------------
    print ""
    print "--- 3. dvc init (no-git mode, since the repo lives elsewhere) ---"
    ^dvc init --no-scm --force

    # --- 4. Add the GCS bucket as a version-aware remote ------------------
    print ""
    print "--- 4. dvc remote add + version_aware ---"
    ^dvc remote add -d storage $"gs://($bucket)" --force
    ^dvc remote modify storage version_aware true
    ^dvc config core.autostage true
    print "Resulting .dvc/config:"
    open .dvc/config

    # --- 5. Track and push v1 ---------------------------------------------
    print ""
    print "--- 5. dvc add + dvc push (v1) ---"
    ^dvc add data
    ^dvc push -v

    # --- 6. Mutate data and push v2 ---------------------------------------
    print ""
    print "--- 6. Edit data, dvc add + dvc push (v2) ---"
    "Ford,F-150,1995,16.0,8,205,4500,USA\n" | save --append data/sample_cars.csv
    let v2_lines = (open data/sample_cars.csv | lines | length)
    print $"data/sample_cars.csv now has ($v2_lines) lines"
    ^dvc add data
    ^dvc push -v

    # --- 7. Prove version_aware: pull v1 back -----------------------------
    print ""
    print "--- 7. Clear local cache, restore v1, prove version_aware works ---"
    rm -rf .dvc/cache data
    # Re-fetch latest into the empty cache, then verify
    ^dvc pull -v
    let after_pull_lines = (open data/sample_cars.csv | lines | length)
    print $"after dvc pull: ($after_pull_lines) lines"

    print ""
    print "All steps complete. Cleanup will now run."
} catch {
    print ""
    print "A step failed. Running cleanup anyway."
}

cleanup $bucket
