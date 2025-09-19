#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?Usage: benchmarks/vllm.sh <BASE_MODEL> [LORA_PATH]}"
LORA_PATH="${2:-}"
LORA_ADAPTER_NAME=""
SIF="$(dirname "$0")/benchmark_container.sif"
PORT=8000

# Build command as an array to avoid passing empty args
CMD=(apptainer exec --nv "$SIF" vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3)

if [[ -n "$LORA_PATH" ]]; then
  ADAPTER_BASENAME="$(basename "$LORA_PATH")"
  LORA_ADAPTER_NAME=$(printf '%s' "$ADAPTER_BASENAME" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/_+/_/g; s/^_+|_+$//g')
  CMD+=(--enable-lora --lora-modules "$LORA_ADAPTER_NAME=$LORA_PATH")
fi

"${CMD[@]}"

