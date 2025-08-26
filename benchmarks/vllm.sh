#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:?Usage: benchmarks/vllm.sh <BASE_MODEL> [LORA_PATH]}"
LORA_PATH="${2:-}"
SIF="$(dirname "$0")/benchmark_container.sif"
PORT=8000
EXTRA=()
[[ -n "$LORA_PATH" ]] && EXTRA+=(--enable-lora --lora-modules "nano=$LORA_PATH")

apptainer exec --nv "$SIF" vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  "${EXTRA[@]:-}"

