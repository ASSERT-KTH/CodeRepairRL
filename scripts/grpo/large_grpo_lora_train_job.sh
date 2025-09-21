#!/bin/bash
#SBATCH --job-name=crrl-large-grpo-lora
#SBATCH --output=logs/large_grpo_lora_%j.out
#SBATCH --error=logs/large_grpo_lora_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 6
#SBATCH --time=48:00:00
#SBATCH -C "fat"


# Large GRPO train job, 6 fat GPUs, 2 running vLLM, 4 training

# Apptainer common runtime configuration (requires CRRL_WORKDIR)
source scripts/appt_common.sh

# MODEL_CONFIG can be provided via env or as --model_config <name>
MODEL_CONFIG="large_qwen"

MASTER_PORT=43001
MODEL_NAME=$(awk -F '"' '/^model_name:/ {print $2; exit}' "src/conf/model/${MODEL_CONFIG}.yaml")

# Context window configuration, this defines our compute requirements more than anything else
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=8192
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))  # not strictly needed, but so we don't get context window errors


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=0,1 crrl.sif \
    trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max_model_len $VLLM_CONTEXT_LENGTH \
    --disable_log_stats \
    --gpu_memory_utilization 0.8 \
    --max_num_seqs 8 \
    --enable_auto_tool_choice \
    --reasoning_parser qwen3 \
    --tool_call_parser hermes \
    --tensor_parallel_size 2 \
    --disable_custom_all_reduce \
    --enforce_eager \
    &  # & makes it run in the background

sleep 200  # give the vLLM server a bit more time to start

apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=2,3,4,5 crrl.sif accelerate launch \
    --main_process_port $MASTER_PORT \
    --config_file scripts/deepspeed/zero3.yaml \
    --num_processes 4 \
    --module src.train_grpo -- \
        run=repo_repair_multilingual \
        model=$MODEL_CONFIG \
        agent.time_limit=80 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_generations=4 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=1 \
        grpo.gradient_accumulation_steps=4 \
        grpo.optim="adamw_torch" \
        "$@"  # pass any additional arguments
    