#!/bin/bash
#SBATCH --job-name=qe_series
#SBATCH --account=def-hanghu
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load quantumespresso/7.5

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Assumes the uploaded directory contains KC60, Na, Nb, V, YBaCuO, and pseudo.
BASE_DIR="${BASE_DIR:-/home/hanghu/links/scratch/EDT}"
PSEUDO_DIR="${PSEUDO_DIR:-${BASE_DIR}/pseudo}"
PREFIX="${PREFIX:-pwscf}"
SLURM_NTASKS="${SLURM_NTASKS:-16}"

# Set RUN_ONLY to a comma-separated list if you want a subset, for example:
#   sbatch --export=ALL,RUN_ONLY=Na,V run_qe_series.sh
RUN_ONLY="${RUN_ONLY:-}"

JOBS=(
  "KC60|KC60.in|KC60PH.in|KC60DM.in"
  "Na|Na-bcc.in|Na-bcc-ph.in|Na-bcc-dm.in"
  "Nb|Nb-cube.in|Nb-cubePH.in|Nb-cubeDM.in"
  "V|V-sp229.in|V-sp229PH.in|V-sp229DM.in"
  "YBaCuO|YBa2CuO7.in|YBa2CuO7-PH.in|YBa2CuO7-DM.in"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

should_run_job() {
  local name="$1"
  [[ -z "$RUN_ONLY" ]] && return 0
  [[ ",$RUN_ONLY," == *",$name,"* ]]
}

check_pseudos() {
  local pw_input="$1"
  local missing=0

  while IFS= read -r pseudo_file; do
    [[ -z "$pseudo_file" ]] && continue
    if [[ ! -f "${PSEUDO_DIR}/${pseudo_file}" ]]; then
      printf 'Missing pseudopotential: %s\n' "${PSEUDO_DIR}/${pseudo_file}" >&2
      missing=1
    fi
  done < <(
    awk '
      {
        line=$0
        sub(/\r$/, "", line)
      }
      line ~ /^[[:space:]]*ATOMIC_SPECIES/ {
        in_species=1
        next
      }
      in_species && line ~ /^[[:space:]]*(ATOMIC_POSITIONS|K_POINTS|CELL_PARAMETERS|CONSTRAINTS|OCCUPATIONS|HUBBARD)/ {
        exit
      }
      in_species && NF >= 3 && $3 ~ /\.UPF$/ {
        print $3
      }
    ' "$pw_input" | sort -u
  )

  [[ "$missing" -eq 0 ]]
}

rewrite_pw_input() {
  local src="$1"
  local dest="$2"
  local outdir="$3"

  awk -v outdir="$outdir" -v pseudo_dir="$PSEUDO_DIR" -v prefix="$PREFIX" '
    BEGIN {
      in_control=0
      saw_outdir=0
      saw_pseudo_dir=0
      saw_prefix=0
    }
    {
      line=$0
      sub(/\r$/, "", line)
      lower=tolower(line)

      if (lower ~ /^[[:space:]]*&control/) {
        in_control=1
      }

      if (in_control && lower ~ /^[[:space:]]*pseudo_dir[[:space:]]*=/) {
        print "    pseudo_dir = \047" pseudo_dir "\047,"
        saw_pseudo_dir=1
        next
      }

      if (in_control && lower ~ /^[[:space:]]*outdir[[:space:]]*=/) {
        print "    outdir = \047" outdir "\047,"
        saw_outdir=1
        next
      }

      if (in_control && lower ~ /^[[:space:]]*prefix[[:space:]]*=/) {
        print "    prefix = \047" prefix "\047,"
        saw_prefix=1
        next
      }

      if (in_control && lower ~ /^[[:space:]]*\//) {
        if (!saw_pseudo_dir) print "    pseudo_dir = \047" pseudo_dir "\047,"
        if (!saw_outdir) print "    outdir = \047" outdir "\047,"
        if (!saw_prefix) print "    prefix = \047" prefix "\047,"
        in_control=0
      }

      print line
    }
  ' "$src" > "$dest"
}

rewrite_ph_input() {
  local src="$1"
  local dest="$2"
  local outdir="$3"

  awk -v outdir="$outdir" -v prefix="$PREFIX" '
    BEGIN {
      in_inputph=0
      saw_outdir=0
      saw_prefix=0
    }
    {
      line=$0
      sub(/\r$/, "", line)
      lower=tolower(line)

      if (lower ~ /^[[:space:]]*&inputph/) {
        in_inputph=1
      }

      if (in_inputph && lower ~ /^[[:space:]]*outdir[[:space:]]*=/) {
        print "  outdir=\047" outdir "\047,"
        saw_outdir=1
        next
      }

      if (in_inputph && lower ~ /^[[:space:]]*prefix[[:space:]]*=/) {
        print "  prefix=\047" prefix "\047,"
        saw_prefix=1
        next
      }

      if (in_inputph && lower ~ /^[[:space:]]*\//) {
        if (!saw_prefix) print "  prefix=\047" prefix "\047,"
        if (!saw_outdir) print "  outdir=\047" outdir "\047,"
        in_inputph=0
      }

      print line
    }
  ' "$src" > "$dest"
}

strip_cr() {
  local src="$1"
  local dest="$2"
  sed 's/\r$//' "$src" > "$dest"
}

run_mpi_qe() {
  local exe="$1"
  local input="$2"
  local output="$3"

  log "Running ${exe} with ${SLURM_NTASKS} MPI tasks"
  mpirun -np "$SLURM_NTASKS" "$exe" -in "$input" > "$output"
}

run_dynmat() {
  local input="$1"
  local output="$2"

  log "Running dynmat.x"
  dynmat.x < "$input" > "$output"
}

run_one_job() {
  local name="$1"
  local pw_file="$2"
  local ph_file="$3"
  local dm_file="$4"
  local job_dir="${BASE_DIR}/${name}"
  local run_dir="${job_dir}/run_${SLURM_JOB_ID:-manual}"
  local qe_tmp="${run_dir}/qe_tmp"

  require_file "${job_dir}/${pw_file}"
  require_file "${job_dir}/${ph_file}"
  require_file "${job_dir}/${dm_file}"
  check_pseudos "${job_dir}/${pw_file}" || die "One or more pseudopotentials are missing for ${name}"

  mkdir -p "$run_dir" "$qe_tmp"

  rewrite_pw_input "${job_dir}/${pw_file}" "${run_dir}/${name}.pw.in" "$qe_tmp"
  rewrite_ph_input "${job_dir}/${ph_file}" "${run_dir}/${name}.ph.in" "$qe_tmp"
  strip_cr "${job_dir}/${dm_file}" "${run_dir}/${name}.dm.in"

  log "Starting ${name}"
  log "Run directory: ${run_dir}"

  (
    cd "$run_dir"
    run_mpi_qe pw.x "${name}.pw.in" "${name}.pw.out"
    run_mpi_qe ph.x "${name}.ph.in" "${name}.ph.out"
    run_dynmat "${name}.dm.in" "${name}.dm.out"
  )

  log "Finished ${name}"
}

[[ -d "$BASE_DIR" ]] || die "BASE_DIR does not exist: $BASE_DIR"
[[ -d "$PSEUDO_DIR" ]] || die "PSEUDO_DIR does not exist: $PSEUDO_DIR"

BASE_DIR="$(cd "$BASE_DIR" && pwd)"
PSEUDO_DIR="$(cd "$PSEUDO_DIR" && pwd)"

log "BASE_DIR=${BASE_DIR}"
log "PSEUDO_DIR=${PSEUDO_DIR}"
log "PREFIX=${PREFIX}"

for job in "${JOBS[@]}"; do
  IFS='|' read -r name pw_file ph_file dm_file <<< "$job"
  if should_run_job "$name"; then
    run_one_job "$name" "$pw_file" "$ph_file" "$dm_file"
  else
    log "Skipping ${name}"
  fi
done

log "All requested QE jobs finished"
