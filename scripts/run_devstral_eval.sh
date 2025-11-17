#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="~/CodeRepairRL"
LITELLM_BIN="$(command -v litellm)"
LITELLM_CONFIG="${ROOT_DIR}/litellm_config.yaml"
LITELLM_PORT="8000"
LITELLM_HOST="0.0.0.0"
LITELLM_BASE="http://${LITELLM_HOST}:${LITELLM_PORT}/v1"

if [[ -z "${LITELLM_BIN}" ]]; then
  echo "litellm CLI not found in PATH. Install litellm before running."
  exit 1
fi

if [[ ! -f "${LITELLM_CONFIG}" ]]; then
  echo "LiteLLM config not found at ${LITELLM_CONFIG}."
  exit 1
fi

if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
  echo "MISTRAL_API_KEY must be exported in the environment before running."
  exit 1
fi

export OPENAI_API_KEY=""
export MISTRAL_API_KEY
export LITELLM_CONFIG

LOG_DIR="${ROOT_DIR}/logs/litellm"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +"%Y%m%d_%H%M%S")_litellm.log"

echo "Starting LiteLLM proxy on ${LITELLM_BASE} using ${LITELLM_CONFIG}"
"${LITELLM_BIN}" --config "${LITELLM_CONFIG}" --port "${LITELLM_PORT}" --debug > "${LOG_FILE}" 2>&1 &
LITELLM_PID=$!

cleanup() {
  if kill -0 "${LITELLM_PID}" > /dev/null 2>&1; then
    echo "Stopping LiteLLM proxy (PID ${LITELLM_PID})"
    kill "${LITELLM_PID}"
    wait "${LITELLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Waiting for LiteLLM proxy to become ready..."
for attempt in {1..30}; do
  if curl --silent --fail "${LITELLM_BASE}/models" > /dev/null 2>&1; then
    echo "LiteLLM proxy is ready."
    break
  fi
  sleep 1
  if [[ ${attempt} -eq 30 ]]; then
    echo "LiteLLM proxy did not become ready in time. Check ${LOG_FILE} for details."
    exit 1
  fi
done

echo "Running nano agent evaluation..."
python "${ROOT_DIR}/benchmarks/swe_bench/run_nano_eval.py" \
  --endpoint "${LITELLM_BASE}" \
  --model-name "openai/devstral-small-2507" \
  --output-dir "${ROOT_DIR}/swe_bench/results_devstral" \
  --subset "verified" \
  --split "test"

echo "nano agent evaluation completed."
echo "LiteLLM logs available at ${LOG_FILE}"

