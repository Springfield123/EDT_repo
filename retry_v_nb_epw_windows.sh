#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
EPW_ROOT="${EPW_ROOT:-${BASE_DIR}/epw_runs}"

cd "$BASE_DIR"

python3 "${SCRIPT_DIR}/scripts/patch_epw_namelist_and_submit_epw_only.py" \
  --epw-root "$EPW_ROOT" \
  --materials V Nb \
  --submit

echo
echo "V/Nb EPW window sanity check:"
for material in V Nb; do
  echo "--- ${material} ---"
  grep -n "nbndsub\\|proj(\\|dis_win\\|dis_froz" "${EPW_ROOT}/${material}/epw.in" || true
done
