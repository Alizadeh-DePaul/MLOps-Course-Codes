#!/usr/bin/env bash
# Exercises/ApiAndRequests/demo.sh - bash end-to-end runner for the API and Requests exercise.
# Run from inside Exercises/ApiAndRequests/ with a clean working tree.
#
# What this does (mirrors the exercise page steps 1-8):
#   1. Sets up a venv and installs deps with uv
#   2. Runs step 1 (status codes)
#   3. Runs step 2 (payloads + GitHub search)
#   4. Runs step 3 (binary download of a PNG)
#   5. Runs step 4 (POST: form vs JSON)
#   6. Sanity-checks the saved img.png and prints a curl equivalent
set -euo pipefail

# --- 1. Environment --------------------------------------------------------
# Install uv once: curl -LsSf https://astral.sh/uv/install.sh | sh
#                  (PowerShell variant on Windows)
uv venv                                    # alt: python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"                 # alt: pip install -e ".[dev]"

# --- 2. Step 1 - status codes ---------------------------------------------
# Sends a 404, a 200, and prints the if/elif/else branching pattern.
python step1_status_codes.py

# --- 3. Step 2 - payloads -------------------------------------------------
# Inspects .content type, parses .json(), hits the GitHub Search API.
# May print a 403 if you've hit the unauthenticated rate limit (10/min).
python step2_payloads.py

# --- 4. Step 3 - binary download ------------------------------------------
# Downloads the PyTorch logo PNG and writes it to ./img.png.
python step3_binary_download.py

# --- 5. Step 4 - POST: form vs JSON ---------------------------------------
# Hits httpbin.org/post twice with the same payload (data= vs json=)
# and prints what the echo server saw.
python step4_post_form_vs_json.py

# --- 6. Sanity check + curl equivalent ------------------------------------
# Confirm img.png landed.
if [ -f img.png ]; then
    bytes=$(wc -c < img.png | tr -d ' ')
    echo "img.png saved: ${bytes} bytes"
else
    echo "img.png missing - step 3 did not write a file"
fi

# Show the curl version of step 1 so students can copy/paste in class.
curl -X GET -I "https://api.github.com"
