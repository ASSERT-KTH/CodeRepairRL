#!/bin/bash
#SBATCH --job-name=crrl-small-sft-lora
#SBATCH --output=logs/small_sft_lora_%j.out
#SBATCH --error=logs/small_sft_lora_%j.err
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"


apptainer exec --nv crrl.sif accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --module src.train_sft -- \
        model=small_qwen \
        model.lora=true \
        model.r=8 \
        model.lora_alpha=16 \
        sft.kl_lambda=0.05 \
        sft.max_length=8192 \
        sft.packing=false \
        sft.per_device_train_batch_size=2 \
        sft.gradient_accumulation_steps=8 \
        "$@"
