#!/usr/bin/env nu
# Exercises/SettingUpGCP/demo.nu - cross-platform smoke test for the GCP setup.
# Run from inside Exercises/SettingUpGCP/. Read-only; does not modify your
# project or auth state. Checks:
#   1. gcloud is on PATH and runs
#   2. There is an active authenticated user
#   3. ADC (application-default credentials) work
#   4. An active project is configured
#   5. The Week 9 / 10 APIs are enabled on that project
$env.config.error_style = "fancy"

let required_apis = [
    "compute.googleapis.com"
    "storage.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "cloudfunctions.googleapis.com"
    "iam.googleapis.com"
    "aiplatform.googleapis.com"
]

mut pass = 0
mut fail = 0

def check [label: string, result: bool, hint: string] {
    if $result {
        print $"  [PASS] ($label)"
    } else {
        print $"  [FAIL] ($label)"
        print $"         fix: ($hint)"
    }
}

print "GCP setup smoke test"
print "===================="

# --- 1. gcloud on PATH -----------------------------------------------------
let has_gcloud = (try { gcloud --version | ignore; true } catch { false })
check "gcloud CLI is installed and on PATH" $has_gcloud "install from https://cloud.google.com/sdk/docs/install, then restart your terminal"
if $has_gcloud { $pass = $pass + 1 } else { $fail = $fail + 1; print "Cannot continue without gcloud. Stopping."; exit 1 }

# --- 2. Active authenticated user -----------------------------------------
let active_account = (try { gcloud auth list --filter=status:ACTIVE --format="value(account)" | str trim } catch { "" })
let has_user = ($active_account | str length) > 0
check $"authenticated as: ($active_account)" $has_user "run: gcloud auth login"
if $has_user { $pass = $pass + 1 } else { $fail = $fail + 1 }

# --- 3. Application Default Credentials -----------------------------------
let has_adc = (try { gcloud auth application-default print-access-token o> /dev/null; true } catch { false })
check "application-default credentials work" $has_adc "run: gcloud auth application-default login"
if $has_adc { $pass = $pass + 1 } else { $fail = $fail + 1 }

# --- 4. Active project ----------------------------------------------------
let active_project = (try { gcloud config get-value project --quiet 2> /dev/null | str trim } catch { "" })
let has_project = (($active_project | str length) > 0) and ($active_project != "(unset)")
check $"active project: ($active_project)" $has_project "run: gcloud config set project <your-project-id>"
if $has_project { $pass = $pass + 1 } else { $fail = $fail + 1 }

# --- 5. Required APIs enabled ---------------------------------------------
if $has_project {
    let enabled = (try { gcloud services list --enabled --format="value(config.name)" | lines } catch { [] })
    for api in $required_apis {
        let is_on = ($api in $enabled)
        check $"API enabled: ($api)" $is_on $"run: gcloud services enable ($api)"
        if $is_on { $pass = $pass + 1 } else { $fail = $fail + 1 }
    }
} else {
    print "  (skipping API checks - no active project)"
}

print ""
print $"Summary: ($pass) passed, ($fail) failed"
if $fail > 0 {
    print "Re-run this script after fixing the failures above."
    exit 1
} else {
    print "All checks passed. Your GCP setup is ready."
}
