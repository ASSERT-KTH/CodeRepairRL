#!/bin/bash
#SBATCH --job-name=swe-mini-eval
#SBATCH --output=logs/swe_mini_%j.out
#SBATCH --error=logs/swe_mini_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=8:00:00
#SBATCH -C "fat"

# SWE-bench evaluation using minisweagent with vLLM server

LORA_PATH=${1:-"/proj/berzelius-2024-336/users/x_bjabj/models/nano_lora"}
BASE_MODEL="Qwen/Qwen3-14B"
PORT=8004  # Unique port for swe_minisweagent

# Launch vLLM server with LoRA adapter
echo "Starting vLLM server with LoRA adapter on port $PORT..."
apptainer exec --nv benchmark_container.sif \
    vllm serve "$BASE_MODEL" \
    --port $PORT \
    --enable-lora \
    --lora-modules "nano=$LORA_PATH" \
    --max-lora-rank 32 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.9 &

VLLM_PID=$!
sleep 30  # Wait for server to start

# Run benchmarks
cd /workspace/benchmarks/swe_bench
mkdir -p results_mini

# Before (baseline)
echo "Running baseline evaluation..."
apptainer exec benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset verified --split dev --slice :25 \
  -m openai/gpt-4o \
  -w 4 \
  -o results_mini/before

# After (LoRA model)
echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy"

apptainer exec benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset verified --split dev --slice :25 \
  -m openai/nano \
  -w 4 \
  -o results_mini/after

# Kill vLLM server
kill $VLLM_PID

# Score with official evaluator
echo "=== Scoring with SWE Bench evaluator ==="
apptainer exec benchmark_container.sif \
    sb-cli submit swe-bench-v dev \
    --predictions_path results_mini/before/preds.json \
    --output_dir results_mini/before_report

apptainer exec benchmark_container.sif \
    sb-cli submit swe-bench-v dev \
    --predictions_path results_mini/after/preds.json \
    --output_dir results_mini/after_report

echo "=== SWE Bench Results ==="
echo "Before report:"
cat results_mini/before_report/report.json | grep "resolved" || echo "No baseline results"
echo "After report:"
cat results_mini/after_report/report.json | grep "resolved" || echo "No LoRA results"