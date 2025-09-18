#!/bin/bash
#SBATCH --job-name=crrl-swe-nano
#SBATCH --output=logs/swe_nano_%A_%a.out
#SBATCH --error=logs/swe_nano_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=08:00:00
#SBATCH -C "fat"
#SBATCH --array=0-9

set -euo pipefail

# Use common Apptainer runtime config (requires CRRL_WORKDIR in env)
source scripts/appt_common.sh

# Defaults
BASE_MODEL="Qwen/Qwen3-8B"     # HF model to serve with vLLM
LORA_PATH=""                    # Optional LoRA path; enables adapter name "nano" if set
MODEL_NAME=""                   # Model name passed to the agent; auto-derived if empty
OUTPUT_DIR="swe_bench/results_nano"
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
      OUTPUT_DIR="${2:?}"; shift 2;;
    --subset)
      SUBSET="${2:?}"; shift 2;;
    --split)
      SPLIT="${2:?}"; shift 2;;
    --slice)
      SLICE="${2:?}"; shift 2;;
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

# Write results to per-shard directory
OUTPUT_DIR="${OUTPUT_DIR}/shard_${TASK_ID}"

ENDPOINT="http://localhost:${PORT}/v1"

# Derive MODEL_NAME if not explicitly provided
if [[ -z "$MODEL_NAME" ]]; then
  if [[ -n "$LORA_PATH" ]]; then
    MODEL_NAME="nano"  # Use LoRA adapter by name
  else
    MODEL_NAME="hosted_vllm/${BASE_MODEL}"
  fi
fi

mkdir -p "$(dirname "logs/.keep")" "$OUTPUT_DIR"

wait_for_vllm() {
  local url="$1"; local -i tries=120
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
    CMD+=(--enable-lora --lora-modules "nano=$LORA_PATH")
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
    --model-name "$MODEL_NAME" \
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


