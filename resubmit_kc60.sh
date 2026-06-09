#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNNER="${RUNNER:-${SCRIPT_DIR}/run_qe_series.sh}"
ACCOUNT="${ACCOUNT:-def-hanghu}"
NTASKS="${NTASKS:-16}"
TIME_LIMIT="${TIME_LIMIT:-72:00:00}"
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
PSEUDO_DIR="${PSEUDO_DIR:-${BASE_DIR}/pseudo}"

[[ -f "$RUNNER" ]] || {
  echo "Missing runner script: $RUNNER" >&2
  echo "Put this script in the same directory as run_qe_series.sh, or set RUNNER=/path/to/run_qe_series.sh" >&2
  exit 1
}

[[ -d "${BASE_DIR}/KC60" ]] || {
  echo "Missing KC60 input directory: ${BASE_DIR}/KC60" >&2
  echo "Set BASE_DIR=/path/to/EDT if your inputs are elsewhere." >&2
  exit 1
}

[[ -d "$PSEUDO_DIR" ]] || {
  echo "Missing pseudopotential directory: $PSEUDO_DIR" >&2
  echo "Set PSEUDO_DIR=/path/to/pseudo if needed." >&2
  exit 1
}

echo "Submitting KC60 retry"
echo "Runner:     $RUNNER"
echo "Base dir:   $BASE_DIR"
echo "Pseudo dir: $PSEUDO_DIR"
echo "Account:    $ACCOUNT"
echo "Tasks:      $NTASKS"
echo "Walltime:   $TIME_LIMIT"

sbatch \
  --job-name=qe_KC60_retry \
  --account="$ACCOUNT" \
  --ntasks="$NTASKS" \
  --time="$TIME_LIMIT" \
  --export=ALL,RUN_ONLY=KC60,BASE_DIR="$BASE_DIR",PSEUDO_DIR="$PSEUDO_DIR" \
  "$RUNNER"
