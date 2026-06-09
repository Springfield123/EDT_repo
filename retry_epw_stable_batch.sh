#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
PSEUDO_DIR="${PSEUDO_DIR:-${BASE_DIR}/pseudo}"
EPW_ROOT="${EPW_ROOT:-${BASE_DIR}/epw_runs}"

cd "$BASE_DIR"

# Stable retry batch for all currently completed non-KC60 materials.
python3 "${SCRIPT_DIR}/scripts/setup_epw_runs.py" \
  --qe-base "$BASE_DIR" \
  --pseudo-dir "$PSEUDO_DIR" \
  --epw-root "$EPW_ROOT" \
  --materials Na V Nb YBaCuO \
  --submit
