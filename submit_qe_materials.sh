#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_qe_series.sh}"

ALL_MATERIALS=(KC60 Na Nb V YBaCuO)

if [[ "$#" -gt 0 ]]; then
  MATERIALS=("$@")
else
  MATERIALS=("${ALL_MATERIALS[@]}")
fi

[[ -f "$RUNNER" ]] || {
  echo "Missing runner script: $RUNNER" >&2
  exit 1
}

for material in "${MATERIALS[@]}"; do
  case " ${ALL_MATERIALS[*]} " in
    *" ${material} "*) ;;
    *)
      echo "Unknown material: ${material}" >&2
      echo "Known materials: ${ALL_MATERIALS[*]}" >&2
      exit 1
      ;;
  esac

  echo "Submitting ${material}"
  sbatch \
    --job-name="qe_${material}" \
    --export=ALL,RUN_ONLY="${material}" \
    "$RUNNER"
done
