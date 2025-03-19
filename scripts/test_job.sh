#!/bin/bash
#SBATCH --job-name=ttc-test
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --gpus 2
#SBATCH --time=0:02:00
#SBATCH -C "thin"
#SBATCH --mail-type=FAIL
#SBATCH --mail-user bhbj@kth.se

# Navigate to the project root directory
cd "$(dirname "$0")/.."

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Directory: $(pwd)"


echo "Python version:"
# The double dash (--) separates uv options from the command to be executed in the uv environment
apptainer exec --nv ttc.sif python --version

echo "CUDA availability:"
apptainer exec --nv ttc.sif python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nv ttc.sif python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
apptainer exec --nv ttc.sif python -c "import torch; print(f'CUDA device name: {torch.cuda.get_device_name(0)}')"

# Test importing key libraries
echo "Testing imports..."
apptainer exec --nv ttc.sif python -c "
import torch
import transformers
import trl
import wandb
import hydra
import datasets
print('All imports successful!')
"

# Test torchrun
echo "Testing torchrun..."
apptainer exec --nv $CONTAINER_IMAGE torchrun --nproc-per-node=1 -m torch.distributed.run --help

# # Clean up
# echo "Cleaning up scratch directory..."
# rm -f "$CONTAINER_IMAGE"
# rmdir "$SCRATCH_DIR" 2>/dev/null || echo "Note: Scratch directory not empty, not removed"

# # Print end time
# echo "End time: $(date)" 