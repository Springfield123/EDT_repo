#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
PSEUDO_DIR="${PSEUDO_DIR:-${BASE_DIR}/pseudo}"
EPW_ROOT="${EPW_ROOT:-${BASE_DIR}/epw_runs}"

cd "$BASE_DIR"

echo "Step 1/2: patch and resubmit EPW-only jobs for Na, V, and Nb"
python3 "${SCRIPT_DIR}/scripts/patch_epw_namelist_and_submit_epw_only.py" \
  --epw-root "$EPW_ROOT" \
  --materials Na V Nb \
  --submit

echo
echo "Step 2/2: regenerate and submit YBaCuO with reduced NSCF settings"
python3 "${SCRIPT_DIR}/scripts/setup_epw_runs.py" \
  --qe-base "$BASE_DIR" \
  --pseudo-dir "$PSEUDO_DIR" \
  --epw-root "$EPW_ROOT" \
  --materials YBaCuO \
  --submit

echo
echo "Generated YBaCuO NSCF sanity check:"
grep -A3 -n "K_POINTS crystal" "${EPW_ROOT}/YBaCuO/nscf.in"
grep -n "nbnd\\|diagonalization\\|electron_maxstep" "${EPW_ROOT}/YBaCuO/nscf.in"

echo
echo "Current EPW manifest rows:"
column -s, -t < "${EPW_ROOT}/epw_run_manifest.csv" || cat "${EPW_ROOT}/epw_run_manifest.csv"
