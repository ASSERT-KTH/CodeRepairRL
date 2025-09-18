#!/usr/bin/env bash

# This file is meant to be sourced from job scripts.
# It requires CRRL_WORKDIR to be set (no directory creation or checks are performed).

if [[ -z "${CRRL_WORKDIR:-}" ]]; then
    echo "[ERROR] CRRL_WORKDIR is not set. Please set it to a writable path, e.g.:" >&2
    echo "     export CRRL_WORKDIR=\"/proj/<project>/users/<user>/\"" >&2
    return 1 2>/dev/null || exit 1
fi

# Reusable common flags for Apptainer execution
# - --nv: GPU access
# - --cleanenv: ignore host env, use only what we pass below
# - --bind: ensure CRRL_WORKDIR path is available inside container at same path
# - --env: set all required environment variables inside container
APPT_COMMON=(
  --nv
  --bind "${CRRL_WORKDIR}:${CRRL_WORKDIR}"
  --env "PROJECT_DIR=${CRRL_WORKDIR}"
  --env "HF_HOME=${CRRL_WORKDIR}/.hf"
  --env "TRANSFORMERS_CACHE=${CRRL_WORKDIR}/.cache/huggingface/transformers"
  --env "HF_DATASETS_CACHE=${CRRL_WORKDIR}/.cache/huggingface/datasets"
  --env "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt"
  --env "PYTHONNOUSERSITE=1"
  --env "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
  --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  # --env "NCCL_DEBUG=INFO"
  # --env "NCCL_DEBUG_SUBSYS=INIT,COLL"
  # --env "NCCL_ASYNC_ERROR_HANDLING=1"
  # --env "TORCH_NCCL_TRACE_BUFFER_SIZE=1048576"
  # --env "TORCH_NCCL_DUMP_ON_TIMEOUT=1"
  # --env "TORCH_NCCL_DESYNC_DEBUG=DETAIL"
  # --env "TORCH_NCCL_BLOCKING_WAIT=1"
)

# Export as a flat string for convenient interpolation in scripts
export APPT_COMMON="${APPT_COMMON[*]}"

echo "Using CRRL_WORKDIR=${CRRL_WORKDIR}"

