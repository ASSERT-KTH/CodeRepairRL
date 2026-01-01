#!/bin/bash
#SBATCH --job-name=pull-swe-images
#SBATCH --output=logs/pull_swe_%A_%a.out
#SBATCH --error=logs/pull_swe_%A_%a.err
#SBATCH --nodes=1
#SBATCH -C "thin"
#SBATCH --gpus 1
#SBATCH --time=06:00:00
#SBATCH --array=0-19

set -euo pipefail

# Configuration
SUBSET="verified"
SPLIT="test"
TOTAL_WORKERS=20
WORKER_ID=${SLURM_ARRAY_TASK_ID:-0}

echo "Starting worker ${WORKER_ID}/${TOTAL_WORKERS} for SWE-bench ${SUBSET}..."

# Use a unique temp file for this job to avoid conflicts
TEMP_SIF="/proj/berzelius-2024-336/users/x_andaf/CodeRepairRL/temp_pull_${SLURM_JOB_ID}_${WORKER_ID}.sif"

# Calculate slice for this worker
# Note: datasets.load_dataset doesn't support stride slicing directly in the load, 
# but we can use Python to select the subset.
# Alternatively, we can update the python script to accept --shard-id and --num-shards.

# Let's update the python command to handle sharding if we modify the script, 
# or use the python script's slice logic if we add it.
# Since the current python script iterates all, we should modify it or use a wrapper.
# The easiest way without modifying python script further is to pass a slice argument.

# Let's modify the python script to accept start/end or shard info. 
# But first, I will write this job script assuming I'll add sharding support to the python script next.

uv run scripts/pull_swe_images.py \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --temp-sif "$TEMP_SIF" \
  --shard-id "$WORKER_ID" \
  --num-shards "$TOTAL_WORKERS"

echo "Worker ${WORKER_ID} finished."

