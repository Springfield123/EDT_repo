#!/bin/bash
#SBATCH --job-name=epw_restart_V
#SBATCH --account=def-hanghu
#SBATCH --ntasks=64
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
export EPW_INPUT="${EPW_INPUT:-epw_restart_narval.in}"
export MPI_LAUNCHER="${MPI_LAUNCHER:-srun}"

if (( EPW_NPOOL * EPW_NIMAGE != SLURM_NTASKS )); then
  echo "EPW parallel layout error: SLURM_NTASKS=$SLURM_NTASKS but EPW_NPOOL*EPW_NIMAGE=$((EPW_NPOOL * EPW_NIMAGE))" >&2
  exit 2
fi

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs outputs save

if ! find . -maxdepth 4 -type f \( -name "*.epb*" -o -name "*.epmatwp*" -o -name "epwdata.fmt" \) | grep -q .; then
  echo "No EPB/EPW restart payload found. Run the first EPW stage to completion or stage epb/epw restart files before submitting this Narval restart." >&2
  exit 3
fi

args=(-npool "$EPW_NPOOL")
if (( EPW_NIMAGE != 1 )); then
  args+=(-nimage "$EPW_NIMAGE")
fi

echo "[$(date '+%F %T')] Running second-stage EPW restart on Narval with $SLURM_NTASKS MPI tasks and ${args[*]}"
if [[ "$MPI_LAUNCHER" == "mpirun" ]]; then
  mpirun -np "$SLURM_NTASKS" epw.x "${args[@]}" -in "$EPW_INPUT" > outputs/epw_restart_narval.out
else
  "$MPI_LAUNCHER" epw.x "${args[@]}" -in "$EPW_INPUT" > outputs/epw_restart_narval.out
fi
echo "[$(date '+%F %T')] EPW Narval restart finished for V"
