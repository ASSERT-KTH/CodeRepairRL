#!/usr/bin/env bash

# This file is meant to be sourced from job scripts.
# It requires CRRL_WORKDIR to be set (no directory creation or checks are performed).

if [[ -z "${CRRL_WORKDIR:-}" ]]; then
    echo "[ERROR] CRRL_WORKDIR is not set. Please set it to a writable path, e.g.:" >&2
    echo "     export CRRL_WORKDIR=\"/proj/<project>/users/<user>/\"" >&2
    return 1 2>/dev/null || exit 1
fi

mkdir -p "${CRRL_WORKDIR}/.tmp/${SLURM_JOB_ID}" \
         "${CRRL_WORKDIR}/.cache/vllm_${SLURM_JOB_ID}"

# Reusable common flags for Apptainer execution
# - --nv: GPU access
# - --cleanenv: ignore host env, use only what we pass below
# - --bind: ensure CRRL_WORKDIR path is available inside container at same path
# - --env: set all required environment variables inside container
APPT_COMMON=(
  --nv
  --cleanenv
  --bind "${CRRL_WORKDIR}:${CRRL_WORKDIR}"
  --home "${CRRL_WORKDIR}"                               # NEW: proper Apptainer home; remove the HOME env override
  --env "PROJECT_DIR=${CRRL_WORKDIR}"
  --env "TMPDIR=${CRRL_WORKDIR}/.tmp/${SLURM_JOB_ID}"    # NEW: job-scoped tmp
  --env "HF_HOME=${CRRL_WORKDIR}/.hf"
  --env "HF_DATASETS_CACHE=${CRRL_WORKDIR}/.hf/datasets"
  # transformers cache is deprecated; safe to remove, but harmless to keep
  --env "TRANSFORMERS_CACHE=${CRRL_WORKDIR}/.cache/huggingface/transformers"
  --env "XDG_CACHE_HOME=${CRRL_WORKDIR}/.cache/${SLURM_JOB_ID}"   # CHANGED: isolate all generic caches per job
  --env "TRITON_CACHE_DIR=${CRRL_WORKDIR}/.cache/triton/${SLURM_JOB_ID}"
  --env "TORCH_HOME=${CRRL_WORKDIR}/.cache/torch/${SLURM_JOB_ID}"
  --env "CUDA_CACHE_PATH=${CRRL_WORKDIR}/.cache/nv/${SLURM_JOB_ID}"
  --env "WANDB_DIR=${CRRL_WORKDIR}/wandb"
  --env "WANDB_CACHE_DIR=${CRRL_WORKDIR}/.cache/wandb/${SLURM_JOB_ID}"
  --env "WANDB_API_KEY=${WANDB_API_KEY}"
  --env "PYTHONNOUSERSITE=1"
  --env "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
  --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=1"

  # Keep these job-scoped (you already did this right)
  --env "TORCHINDUCTOR_CACHE_DIR=${CRRL_WORKDIR}/.cache/torchinductor_${SLURM_JOB_ID}"
  --env "PYTORCH_KERNEL_CACHE_PATH=${CRRL_WORKDIR}/.cache/pytorch_kernel_${SLURM_JOB_ID}"

  # vLLM has its own cache root; isolate it per job explicitly
  --env "VLLM_CACHE_DIR=${CRRL_WORKDIR}/.cache/vllm_${SLURM_JOB_ID}"   # NEW: prevents cross-job reuse

  # Nice-to-haves
  --env "TOKENIZERS_PARALLELISM=false"
  # Debug toggles (enable only when diagnosing):
  # --env "CUDA_LAUNCH_BLOCKING=1"
)

# Export as a flat string for convenient interpolation in scripts
export APPT_COMMON="${APPT_COMMON[*]}"

echo "Using CRRL_WORKDIR=${CRRL_WORKDIR}"

