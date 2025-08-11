#!/bin/bash
#SBATCH --job-name=swe-nano-eval
#SBATCH --output=logs/swe_nano_%j.out
#SBATCH --error=logs/swe_nano_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=8:00:00
#SBATCH -C "fat"

# SWE-bench evaluation using nano_agent with vLLM server

LORA_PATH=${1:-"/proj/berzelius-2024-336/users/x_bjabj/models/nano_lora"}
BASE_MODEL="Qwen/Qwen3-14B"
PORT=8003  # Unique port for swe_nano

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
mkdir -p results_nano

# Run nano_agent evaluation
echo "Running nano_agent evaluation..."
apptainer exec benchmark_container.sif python run_nano_eval.py \
    --endpoint "http://localhost:$PORT/v1" \
    --model-name "nano" \
    --output-dir results_nano \
    --subset verified \
    --split dev \
    --slice :25

# Kill vLLM server
kill $VLLM_PID

# Submit predictions to SWE-bench
echo "=== Scoring with SWE Bench evaluator ==="
apptainer exec benchmark_container.sif \
    sb-cli submit swe-bench-v dev \
    --predictions_path results_nano/preds.json \
    --output_dir results_nano/report

echo "=== SWE Bench Results ==="
cat results_nano/report/report.json | grep "resolved" || echo "No results found"