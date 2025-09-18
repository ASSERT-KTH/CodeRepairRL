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

# MODEL_CONFIG can be provided via env or as --model_config <name>
MODEL_CONFIG="${MODEL_CONFIG:-medium_qwen}"
if [[ "${1:-}" == --model_config=* ]]; then MODEL_CONFIG="${1#*=}"; shift; fi
if [[ "${1:-}" == --model_config ]]; then MODEL_CONFIG="${2:?}"; shift 2; fi


MASTER_PORT=43001
MODEL_NAME=$(awk -F '"' '/^model_name:/ {print $2; exit}' "src/conf/model/${MODEL_CONFIG}.yaml")

RP=""; TP=""; CT=""; PLUG=""
case "${MODEL_CONFIG,,}" in
  *qwen*)     RP="--reasoning_parser qwen3"; TP="--tool_call_parser hermes";;
  # xlam isn't the actual parser we use, is a workaround for a vLLM bug that validates tool-call-parser before injecting our one, so we override
  *nemotron*) TP="--tool_call_parser llama3_nemotron_json"; CT="--chat-template src/chat_templates/llama_nemotron_nano_generic_tool_calling.jinja"; PLUG="--tool_parser_plugin src/chat_templates/llama_nemotron_nano_toolcall_parser.py";;
  *llama*)    TP="--tool_call_parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *mistral*)  TP="--tool_call_parser mistral"; CT="--chat-template src/chat_templates/tool_chat_template_mistral.jinja";;
  *)          TP="--tool_call_parser hermes";;
esac

echo "CT: $CT"
echo "PLUG: $PLUG"
echo "RP: $RP"
echo "TP: $TP"


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
    $CT \
    $PLUG \
    $RP $TP \
    &


apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=1,2 crrl.sif accelerate launch \
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
        "$@"  # pass any additional arguments
