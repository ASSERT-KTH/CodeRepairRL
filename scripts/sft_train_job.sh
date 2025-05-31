#!/bin/bash
#SBATCH --job-name=swe-gym-sft
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#SBATCH --gpus 1
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# SFT training with DeepSpeed on curated SWE-Gym data
# Uses 1 GPUs for training


# Training parameters
DATASET_NAME="bjarni/swe-gym-lite-sft"
MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_MODEL="bjarni/qwen3-8b-swe-gym-sft"
BATCH_SIZE=4
GRAD_ACCUM=4
EPOCHS=3

echo "Starting SFT training job"
echo "Dataset: $DATASET_NAME"
echo "Base model: $MODEL_NAME"
echo "Output model: $OUTPUT_MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Epochs: $EPOCHS"

# Create logs directory if it doesn't exist
mkdir -p logs

# Check that HuggingFace token is available
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    exit 1
fi

# Run SFT training with DeepSpeed using Hydra config overrides
CUDA_VISIBLE_DEVICES=0 FLASH_ATTENTION_FORCE_DISABLED=1 apptainer exec --nv crrl.sif \
    python -m src.train_sft \
    run.wandb_project="SWE-Gym-SFT-Production" \
    run.dataset_name="$DATASET_NAME" \
    run.output_model_name="$OUTPUT_MODEL" \
    model.model_name="$MODEL_NAME" \
    sft.per_device_train_batch_size=$BATCH_SIZE \
    sft.gradient_accumulation_steps=$GRAD_ACCUM \
    sft.num_train_epochs=$EPOCHS \
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