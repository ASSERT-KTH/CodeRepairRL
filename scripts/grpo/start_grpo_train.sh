#!/bin/bash

# Start GRPO training
# Usage: ./start_grpo_train.sh [--model_config <name>] [additional args...]

set -e

# MODEL_CONFIG can be provided via env or as --model_config <name>
MODEL_CONFIG="${MODEL_CONFIG:-medium_qwen}"
if [[ "${1:-}" == --model_config=* ]]; then MODEL_CONFIG="${1#*=}"; shift; fi
if [[ "${1:-}" == --model_config ]]; then MODEL_CONFIG="${2:?}"; shift 2; fi

MASTER_PORT=43001

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=10240

echo "Starting GRPO training"
echo "Model Config: $MODEL_CONFIG"
echo "Master Port: $MASTER_PORT"
echo ""

CUDA_VISIBLE_DEVICES=1,2 PYTHONNOUSERSITE=1 uv run accelerate launch \
    --main_process_port $MASTER_PORT \
    --num_processes 2 \
    --config_file scripts/deepspeed/zero2.yaml \
    --module src.train_grpo -- \
        run=repo_repair_multilingual \
        model=$MODEL_CONFIG \
        agent.time_limit=60 \
        grpo=multi_turn_gspo \
        grpo.max_prompt_length=$MAX_PROMPT_LENGTH \
        grpo.max_completion_length=$MAX_COMPLETION_LENGTH \
        grpo.num_generations=8 \
        grpo.generation_batch_size=8 \
        grpo.per_device_train_batch_size=4 \
        grpo.gradient_accumulation_steps=4 \
        grpo.optim="adamw_torch" \
        "$@"

