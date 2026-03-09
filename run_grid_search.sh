#!/bin/bash
#SBATCH --job-name=qml-grid
#SBATCH --array=1-4
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1024M
#SBATCH --time=00:30:00
#SBATCH --partition=compute
# ============================================================================
# Distributed QML Grid Search — Slurm Job Array Launcher
#
# Each array task pulls its hyperparameters from params.csv using the
# SLURM_ARRAY_TASK_ID, then runs the Dockerized VQC training.
#
# Submit with:   sbatch run_grid_search.sh
# Monitor with:  squeue -u $USER
# ============================================================================

set -euo pipefail

IMAGE_NAME="qml-grid:latest"
PARAM_FILE="params.csv"
RESULT_DIR="$(pwd)/results"

echo "========================================"
echo " Job Array ID  : ${SLURM_ARRAY_JOB_ID}"
echo " Task ID       : ${SLURM_ARRAY_TASK_ID}"
echo " Hostname      : $(hostname)"
echo " Date          : $(date -Iseconds)"
echo "========================================"

# Ensure the results directory exists on this node
mkdir -p "${RESULT_DIR}"

# Run the containerized training.
# --rm          : clean up container after exit
# -v            : mount results dir so output persists on the host
# --cpus        : limit container to allocated CPUs
# --memory      : limit container to allocated memory
docker run --rm \
    -v "${RESULT_DIR}:/app/results" \
    --cpus="1" \
    --memory="896m" \
    "${IMAGE_NAME}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --param-file params.csv \
    --output-dir results

EXIT_CODE=$?

echo "========================================"
echo " Task ${SLURM_ARRAY_TASK_ID} finished with exit code ${EXIT_CODE}"
echo " Completed at: $(date -Iseconds)"
echo "========================================"

exit ${EXIT_CODE}
