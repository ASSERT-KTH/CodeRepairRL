#!/bin/bash
#SBATCH --job-name=crrl-large-sft-lora
#SBATCH --output=logs/large_sft_lora_%j.out
#SBATCH --error=logs/large_sft_lora_%j.err
#SBATCH --gpus 4
#SBATCH --time=72:00:00
#SBATCH -C "fat"

apptainer exec --nv crrl.sif accelerate launch \
    --config_file scripts/deepspeed/zero3.yaml \
    --num_processes 4 \
    --module src.train_sft -- \
        model=large_qwen \
        model.lora=true \
        sft.learning_rate=1e-4 \
        sft.max_length=12288 \
        sft.per_device_train_batch_size=1 \
        sft.gradient_accumulation_steps=8 \
        sft.packing=false \
        sft.ddp_bucket_cap_mb=16 \
        sft.ddp_find_unused_parameters=false \
        "$@"