#!/bin/bash

# Start vLLM server for GRPO training
# Usage: ./start_vllm_server.sh [--model_config <name>]

set -e

# MODEL_CONFIG can be provided via env or as --model_config <name>
MODEL_CONFIG="${MODEL_CONFIG:-medium_qwen}"
if [[ "${1:-}" == --model_config=* ]]; then MODEL_CONFIG="${1#*=}"; shift; fi
if [[ "${1:-}" == --model_config ]]; then MODEL_CONFIG="${2:?}"; shift 2; fi

MODEL_NAME=$(awk -F '"' '/^model_name:/ {print $2; exit}' "src/conf/model/${MODEL_CONFIG}.yaml")

# Parser configuration based on model type
RP=""; TP=""; CT=""
case "${MODEL_CONFIG,,}" in
  *qwen*)     RP="--reasoning_parser qwen3"; TP="--tool_call_parser hermes";;
  *nemotron*) TP="--tool_call_parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *llama*)    TP="--tool_call_parser llama3_json"; CT="--chat-template src/chat_templates/tool_chat_template_llama3.1_json.jinja";;
  *mistral*)  TP="--tool_call_parser hermes";;
  *)          TP="--tool_call_parser hermes";;
esac

# Context window configuration
MAX_PROMPT_LENGTH=1024
MAX_COMPLETION_LENGTH=10240
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_COMPLETION_LENGTH))
VLLM_CONTEXT_LENGTH=$((MAX_CONTEXT_LENGTH + 1024))

echo "Starting vLLM server"
echo "Model Config: $MODEL_CONFIG"
echo "Model Name: $MODEL_NAME"
echo ""

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 uv run trl vllm-serve-async \
    --model "$MODEL_NAME" \
    --max-model-len $VLLM_CONTEXT_LENGTH \
    --disable-log-stats \
    --gpu-memory-utilization 0.94 \
    --enable-auto-tool-choice \
    $CT \
    $RP $TP

