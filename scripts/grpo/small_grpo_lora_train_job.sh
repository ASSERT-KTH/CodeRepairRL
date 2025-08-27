#!/bin/bash
#SBATCH --job-name=crrl-small-grpo-lora
#SBATCH --output=logs/small_grpo_lora_%j.out
#SBATCH --error=logs/small_grpo_lora_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 2
#SBATCH --time=24:00:00
#SBATCH -C "fat"


# Small GRPO train job, 2 fat GPUs, 1 running vLLM, 1 training

# Apptainer common runtime configuration (requires CRRL_WORKDIR)
source scripts/appt_common.sh


MASTER_PORT=43001
MODEL_CONFIG="small_qwen"
MODEL_NAME=$(grep -Po '^model_name: "\K[^"]*' src/conf/model/${MODEL_CONFIG}.yaml)

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=12288
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=0 crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max-model-len $VLLM_CONTEXT_LENGTH \
    --disable_log_stats \
    --gpu-memory-utilization 0.94 \
    --enable_auto_tool_choice \
    --reasoning_parser qwen3 \
    --tool_call_parser hermes \
    &


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=1 crrl.sif accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_processes 1 \
    --module src.train_grpo -- \
        run=repo_repair \
        run.dataset_name="SWE-Gym/SWE-Gym-Lite" \
        model=$MODEL_CONFIG \
        model.model_name=$MODEL_NAME \
        agent.time_limit=60 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_generations=4 \
        grpo.steps_per_generation=3 \
        grpo.per_device_train_batch_size=4 \
        grpo.gradient_accumulation_steps=3 \
        grpo.optim="paged_adamw_8bit" \
        "$@"  # pass any additional arguments
