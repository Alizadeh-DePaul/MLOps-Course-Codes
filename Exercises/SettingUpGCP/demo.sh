#!/usr/bin/env bash
# Exercises/SettingUpGCP/demo.sh - bash smoke test for the GCP setup.
# Run from inside Exercises/SettingUpGCP/. Read-only; does not modify your
# project or auth state. Checks:
#   1. gcloud is on PATH and runs
#   2. There is an active authenticated user
#   3. ADC (application-default credentials) work
#   4. An active project is configured
#   5. The Week 9 / 10 APIs are enabled on that project
set -uo pipefail

REQUIRED_APIS=(
    "compute.googleapis.com"
    "storage.googleapis.com"
    "artifactregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "cloudfunctions.googleapis.com"
    "iam.googleapis.com"
    "aiplatform.googleapis.com"
)

PASS=0
FAIL=0

check() {
    local label="$1"
    local ok="$2"
    local hint="$3"
    if [ "$ok" = "1" ]; then
        echo "  [PASS] $label"
        PASS=$((PASS + 1))
    else
        echo "  [FAIL] $label"
        echo "         fix: $hint"
        FAIL=$((FAIL + 1))
    fi
}

echo "GCP setup smoke test"
echo "===================="

# --- 1. gcloud on PATH -----------------------------------------------------
if command -v gcloud >/dev/null 2>&1; then
    check "gcloud CLI is installed and on PATH" 1 ""
else
    check "gcloud CLI is installed and on PATH" 0 "install from https://cloud.google.com/sdk/docs/install, then restart your terminal"
    echo "Cannot continue without gcloud. Stopping."
    exit 1
fi

# --- 2. Active authenticated user -----------------------------------------
ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | tr -d '[:space:]')"
if [ -n "$ACTIVE_ACCOUNT" ]; then
    check "authenticated as: $ACTIVE_ACCOUNT" 1 ""
else
    check "authenticated as: (none)" 0 "run: gcloud auth login"
fi

# --- 3. Application Default Credentials -----------------------------------
if gcloud auth application-default print-access-token >/dev/null 2>&1; then
    check "application-default credentials work" 1 ""
else
    check "application-default credentials work" 0 "run: gcloud auth application-default login"
fi

# --- 4. Active project ----------------------------------------------------
ACTIVE_PROJECT="$(gcloud config get-value project --quiet 2>/dev/null | tr -d '[:space:]')"
if [ -n "$ACTIVE_PROJECT" ] && [ "$ACTIVE_PROJECT" != "(unset)" ]; then
    check "active project: $ACTIVE_PROJECT" 1 ""
    HAS_PROJECT=1
else
    check "active project: (unset)" 0 "run: gcloud config set project <your-project-id>"
    HAS_PROJECT=0
fi

# --- 5. Required APIs enabled ---------------------------------------------
if [ "$HAS_PROJECT" = "1" ]; then
    ENABLED="$(gcloud services list --enabled --format='value(config.name)' 2>/dev/null)"
    for api in "${REQUIRED_APIS[@]}"; do
        if echo "$ENABLED" | grep -qx "$api"; then
            check "API enabled: $api" 1 ""
        else
            check "API enabled: $api" 0 "run: gcloud services enable $api"
        fi
    done
else
    echo "  (skipping API checks - no active project)"
fi

echo ""
echo "Summary: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
    echo "Re-run this script after fixing the failures above."
    exit 1
else
    echo "All checks passed. Your GCP setup is ready."
fi
