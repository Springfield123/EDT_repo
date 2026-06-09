#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
PSEUDO_DIR="${PSEUDO_DIR:-${BASE_DIR}/pseudo}"
EPW_ROOT="${EPW_ROOT:-${BASE_DIR}/epw_runs}"

cd "$BASE_DIR"

echo "Step 1/2: patch and submit EPW-only retries for V and Nb"
python3 "${SCRIPT_DIR}/scripts/patch_epw_namelist_and_submit_epw_only.py" \
  --epw-root "$EPW_ROOT" \
  --materials V Nb \
  --submit

echo
echo "Step 2/2: regenerate and submit full corrected YBaCuO run"
python3 "${SCRIPT_DIR}/scripts/setup_epw_runs.py" \
  --qe-base "$BASE_DIR" \
  --pseudo-dir "$PSEUDO_DIR" \
  --epw-root "$EPW_ROOT" \
  --materials YBaCuO \
  --submit

echo
echo "V/Nb EPW sanity check:"
for material in V Nb; do
  echo "--- ${material} ---"
  grep -n "nbndsub\\|proj(\\|dis_win\\|dis_froz\\|kmaps\\|fildyn" "${EPW_ROOT}/${material}/epw.in" || true
done

echo
echo "YBaCuO NSCF sanity check:"
grep -A3 -n "K_POINTS crystal" "${EPW_ROOT}/YBaCuO/nscf.in"
grep -n "nbnd\\|diagonalization\\|electron_maxstep" "${EPW_ROOT}/YBaCuO/nscf.in"

echo
echo "YBaCuO EPW sanity check:"
grep -n "nbndsub\\|proj(\\|dis_win\\|dis_froz\\|kmaps\\|fildyn" "${EPW_ROOT}/YBaCuO/epw.in" || true
