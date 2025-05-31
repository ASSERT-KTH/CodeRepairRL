#!/bin/bash
#SBATCH --job-name=swe-gym-sft
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --gpus 1
#SBATCH --time=24:00:00
#SBATCH -C "fat"


# Check that HuggingFace token is available
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 FLASH_ATTENTION_FORCE_DISABLED=1 apptainer exec --nv crrl.sif \
    python -m src.train_sft \
    run.push_to_hub=true \
    "$@"  # Pass any additional Hydra overrides

SFT_EXIT_CODE=$?

if [ $SFT_EXIT_CODE -eq 0 ]; then
    echo "SFT training completed successfully!"
    echo "Model saved as: $OUTPUT_MODEL"
else
    echo "SFT training failed with exit code: $SFT_EXIT_CODE"
    exit $SFT_EXIT_CODE
fi