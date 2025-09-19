#!/bin/bash
#SBATCH --job-name=crrl-swe-nano
#SBATCH --output=logs/swe_nano_%A_%a.out
#SBATCH --error=logs/swe_nano_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=00:30:00
#SBATCH -C "fat"
#SBATCH --array=0-9

set -euo pipefail

# Use common Apptainer runtime config (requires CRRL_WORKDIR in env)
source scripts/appt_common.sh

# Defaults
BASE_MODEL="Qwen/Qwen3-8B"     # HF model to serve with vLLM
LORA_PATH=""                    # Optional LoRA path; adapter name auto-derived from basename if set
MODEL_NAME=""                   # Model name passed to the agent; auto-derived if empty
SCAFFOLD="nano-agent"           # Scaffold identifier for run tagging
OUTPUT_BASE_DIR="swe_bench/results"
SUBSET="verified"
SPLIT="test"
SLICE=""
PORT=8000
SIF="benchmarks/benchmark_container.sif"
START_SERVER=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="${2:?}"; shift 2;;
    --lora-path)
      LORA_PATH="${2:?}"; shift 2;;
    --model-name)
      MODEL_NAME="${2:?}"; shift 2;;
    --output-dir)
      OUTPUT_BASE_DIR="${2:?}"; shift 2;;
    --subset)
      SUBSET="${2:?}"; shift 2;;
    --split)
      SPLIT="${2:?}"; shift 2;;
    --slice)
      SLICE="${2:?}"; shift 2;;
    --scaffold)
      SCAFFOLD="${2:?}"; shift 2;;
    --port)
      PORT="${2:?}"; shift 2;;
    --no-server)
      START_SERVER=0; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# Derive slice and per-task settings when running as a SLURM array
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SHARD_SIZE=50

# Auto-compute slice if not explicitly provided
if [[ -z "$SLICE" ]]; then
  START=$(( TASK_ID * SHARD_SIZE ))
  END=$(( START + SHARD_SIZE ))
  SLICE="${START}:${END}"
fi

# Offset port to avoid conflicts if multiple tasks land on the same node
if [[ $START_SERVER -eq 1 ]]; then
  PORT=$(( PORT + TASK_ID ))
fi

ENDPOINT="http://localhost:${PORT}/v1"

# Derive MODEL_NAME and LoRA adapter name if not explicitly provided
LORA_ADAPTER_NAME=""
if [[ -n "$LORA_PATH" ]]; then
  if [[ -z "$MODEL_NAME" ]]; then
    # Derive adapter name from LoRA path basename and sanitize for use as model id
    ADAPTER_BASENAME="$(basename "$LORA_PATH")"
    LORA_ADAPTER_NAME=$(printf '%s' "$ADAPTER_BASENAME" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
    MODEL_NAME="$LORA_ADAPTER_NAME"
  else
    # Honor explicit model name; use it as the LoRA adapter name
    LORA_ADAPTER_NAME="$MODEL_NAME"
  fi
else
  # No LoRA: default model name is the base model being served
  if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$BASE_MODEL"
  fi
fi

# Build a descriptive run tag: <scaffold>-<model_tag>
sanitize_tag() {
  local s="$1"
  s="${s//\//__}"
  s="${s// /_}"
  s=$(printf '%s' "$s" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
  printf '%s' "$s"
}

if [[ -n "$LORA_PATH" ]]; then
  BASE_TAG=$(sanitize_tag "$BASE_MODEL")
  ADAPTER_TAG=$(sanitize_tag "$(basename "$LORA_PATH")")
  MODEL_TAG="${BASE_TAG}__lora__${ADAPTER_TAG}"
else
  MODEL_TAG=$(sanitize_tag "$MODEL_NAME")
fi

RUN_TAG="${SCAFFOLD}-${MODEL_TAG}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${RUN_TAG}/shard_${TASK_ID}"

mkdir -p "$(dirname "logs/.keep")" "$OUTPUT_DIR"

wait_for_vllm() {
  local url="$1"; local -i tries=180
  while (( tries-- > 0 )); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$url/models" || true)
    if [[ "$code" == "200" ]]; then return 0; fi
    sleep 2
  done
  return 1
}

VLLM_PID=""
if [[ $START_SERVER -eq 1 ]]; then
  echo "Starting vLLM server on port $PORT for base model '$BASE_MODEL'..."
  CMD=(apptainer exec $APPT_COMMON --env CUDA_VISIBLE_DEVICES=0 "$SIF" vllm serve "$BASE_MODEL" \
    --port "$PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3)

  if [[ -n "$LORA_PATH" ]]; then
    CMD+=(--max-lora-rank 32 --enable-lora --lora-modules "$LORA_ADAPTER_NAME=$LORA_PATH")
  fi

  # Start server in background and capture PID
  "${CMD[@]}" > "logs/vllm_${SLURM_JOB_ID:-$$}.log" 2>&1 &
  VLLM_PID=$!
  trap 'if [[ -n "$VLLM_PID" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi' EXIT

  echo "Waiting for vLLM to become ready at $ENDPOINT ..."
  if ! wait_for_vllm "$ENDPOINT"; then
    echo "vLLM did not become ready in time" >&2
    exit 1
  fi
fi

echo "Running nano_agent evaluation with model '$MODEL_NAME'..."
apptainer exec $APPT_COMMON \
  --env OPENAI_API_BASE="$ENDPOINT" \
  --env OPENAI_API_KEY="dummy" \
  "$SIF" python3 benchmarks/swe_bench/run_nano_eval.py \
    --endpoint "$ENDPOINT" \
    --model-name "hosted_vllm/$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --subset "$SUBSET" \
    --split "$SPLIT" \
    --slice "$SLICE"

echo "Predictions saved to $OUTPUT_DIR/preds.jsonl"

# Stop vLLM if we started it
if [[ -n "$VLLM_PID" ]]; then
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
fi


