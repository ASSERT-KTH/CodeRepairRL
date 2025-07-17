#!/bin/bash
#SBATCH --job-name=crrl-large-grpo-lora
#SBATCH --output=logs/large_grpo_lora_%j.out
#SBATCH --error=logs/large_grpo_lora_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 6
#SBATCH --time=48:00:00
#SBATCH -C "fat"

# Large GRPO training job, 4 GPUs, 2 running vLLM, 2 training

# This was crucial to find errors when running distributed training, i.e. quit on deadlock instead of hanging
export NCCL_ASYNC_ERROR_HANDLING=1
MASTER_ADDR=$(hostname -s)
MASTER_PORT=43001

# Model configuration - use merged SFT model for simplified VLLM pipeline
MODEL_CONFIG="large_qwen"
MODEL_NAME=$(grep -Po '^model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration, this defines our compute requirements more than anything else
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=12288
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors

# VLLM server - loads initial model (any same-architecture model works)
# Training server will sync weights from training model before first inference
CUDA_VISIBLE_DEVICES=4,5 apptainer exec --nv crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $VLLM_CONTEXT_LENGTH \
    --disable_log_stats \
    --gpu_memory_utilization 0.94 \
    --enable_auto_tool_choice \
    --reasoning_parser qwen3 \
    --tool_call_parser hermes \
    --tensor_parallel_size 2 \
    &  # & makes it run in the background

sleep 200  # wait for vLLM server to start

# IMPORTANT: train job should include DEVICE 0
CUDA_VISIBLE_DEVICES=0,1,2,3 apptainer exec --nv crrl.sif accelerate launch \
    --config_file scripts/deepspeed/zero3.yaml \
    --num_processes 4 \
    --module src.train_grpo -- \
        run=repo_repair \
        run.dataset_name="SWE-Gym/SWE-Gym" \
        model=$MODEL_CONFIG \
        model.model_name=$MODEL_NAME \
        grpo=multi_turn \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_train_epochs=2 \
        grpo.num_generations=8 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=2 \
        grpo.gradient_accumulation_steps=8 \
        grpo.beta=0.04 \
        grpo.scale_rewards=false \
        grpo.loss_type=grpo \
        grpo.optim="adamw_torch" \
        "$@"  # pass any additional arguments
    