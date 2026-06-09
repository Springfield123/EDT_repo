#!/bin/bash
#SBATCH --job-name=epw_only_YBaCuO
#SBATCH --account=def-hanghu
#SBATCH --ntasks=48
#SBATCH --time=72:00:00
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

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs save

args=(-npool "$EPW_NPOOL")
if (( EPW_NIMAGE != 1 )); then
  args+=(-nimage "$EPW_NIMAGE")
fi

echo "[$(date '+%F %T')] Running epw.x only with $SLURM_NTASKS MPI tasks and ${args[*]}"
mpirun -np "$SLURM_NTASKS" epw.x "${args[@]}" -in epw.in > outputs/epw.out
echo "[$(date '+%F %T')] EPW-only retry finished for YBaCuO"
