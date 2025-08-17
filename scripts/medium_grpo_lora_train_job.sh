#!/bin/bash
#SBATCH --job-name=crrl-medium-grpo-lora
#SBATCH --output=logs/medium_grpo_lora_%j.out
#SBATCH --error=logs/medium_grpo_lora_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 3
#SBATCH --time=24:00:00
#SBATCH -C "fat"

# Medium GRPO train job, 3 fat GPUs, 1 running vLLM, 2 training

# Apptainer common runtime configuration (requires CRRL_WORKDIR)
source scripts/appt_common.sh


# This was crucial to find errors when running distributed training, i.e. quit on deadlock instead of hanging
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
MASTER_ADDR=$(hostname -s)
MASTER_PORT=43001

# Model configuration - use merged SFT model for simplified VLLM pipeline
MODEL_CONFIG="medium_qwen"
MODEL_NAME=$(grep -Po '^model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)
# MODEL_NAME="ASSERT-KTH/Qwen3-8B-Nano-SWE-Gym-SFT"

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=12288
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=0 crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max-model-len $VLLM_CONTEXT_LENGTH \
    --gpu-memory-utilization 0.94 \
    --async-scheduling \
    --enable-prefix-caching \
    --max-num-seqs 32 \
    --max-num-batched-tokens 8192 \
    --long-prefill-token-threshold 2048 \
    --disable_log_stats \
    --enable_auto_tool_choice \
    --reasoning_parser qwen3 \
    --tool_call_parser hermes \
    &


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=1,2 crrl.sif accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_processes 2 \
    --config_file scripts/deepspeed/zero2.yaml \
    --module src.train_grpo -- \
        run=repo_repair \
        model=$MODEL_CONFIG \
        model.model_name=$MODEL_NAME \
        agent.time_limit=60 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_train_epochs=10 \
        grpo.num_generations=8 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=4 \
        grpo.gradient_accumulation_steps=4 \
        grpo.optim="adamw_torch" \
        "$@"  # pass any additional arguments
