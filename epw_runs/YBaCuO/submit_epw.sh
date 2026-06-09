#!/bin/bash
#SBATCH --job-name=epw_YBaCuO
#SBATCH --account=def-hanghu
#SBATCH --ntasks=48
#SBATCH --time=96:00:00
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load quantumespresso/7.5

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export EPW_NIMAGE="${EPW_NIMAGE:-1}"
export EPW_NPOOL="${EPW_NPOOL:-$SLURM_NTASKS}"
export CLEAN_QE_OUT="${CLEAN_QE_OUT:-1}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  echo "Set EPW_NPOOL and EPW_NIMAGE so their product equals SLURM_NTASKS." >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
if [[ "$CLEAN_QE_OUT" == "1" ]]; then
  echo "[$(date '+%F %T')] Cleaning local retry artifacts in $SLURM_SUBMIT_DIR"
  rm -rf qe_out tmp
  rm -f outputs/scf.out outputs/nscf.out outputs/epw.out
fi
mkdir -p logs outputs qe_out tmp

run_mpi() {
  local exe="$1"
  local input="$2"
  local output="$3"
  echo "[$(date '+%F %T')] Running $exe on $SLURM_NTASKS MPI tasks"
  mpirun -np "$SLURM_NTASKS" "$exe" -in "$input" > "$output"
}

run_epw() {
  local input="$1"
  local output="$2"
  local args=(-npool "$EPW_NPOOL")
  if (( EPW_NIMAGE != 1 )); then
    args+=(-nimage "$EPW_NIMAGE")
  fi
  echo "[$(date '+%F %T')] Running epw.x on $SLURM_NTASKS MPI tasks with ${args[*]}"
  mpirun -np "$SLURM_NTASKS" epw.x "${args[@]}" -in "$input" > "$output"
}

run_mpi pw.x scf.in outputs/scf.out
run_mpi pw.x nscf.in outputs/nscf.out
run_epw epw.in outputs/epw.out

echo "[$(date '+%F %T')] EPW workflow finished for YBaCuO"
