#!/bin/bash
#SBATCH --job-name=terminal-bench-eval
#SBATCH --output=logs/terminal_bench_%j.out
#SBATCH --error=logs/terminal_bench_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=4:00:00
#SBATCH -C "fat"

# Terminal Bench evaluation with vLLM server

LORA_PATH=${1:-"/proj/berzelius-2024-336/users/x_bjabj/models/nano_lora"}
BASE_MODEL="Qwen/Qwen3-14B"
PORT=8002  # Unique port for terminal_bench

# Launch vLLM server with LoRA adapter
echo "Starting vLLM server with LoRA adapter on port $PORT..."
apptainer exec --nv benchmark_container.sif \
    vllm serve "$BASE_MODEL" \
    --port $PORT \
    --enable-lora \
    --lora-modules "nano=$LORA_PATH" \
    --max-lora-rank 32 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 &

VLLM_PID=$!
sleep 30  # Wait for server to start

# Run benchmarks
cd /workspace/benchmarks/terminal_bench
mkdir -p results

# Before (baseline)
echo "Running baseline evaluation..."
apptainer exec benchmark_container.sif tb run \
  --agent terminus \
  --model-name openai/gpt-4o \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  --n-concurrent 4 \
  --output-dir results/before

# After (LoRA model)
echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy"

apptainer exec benchmark_container.sif tb run \
  --agent terminus \
  --model-name openai/nano \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  --n-concurrent 4 \
  --output-dir results/after

# Kill vLLM server
kill $VLLM_PID

# Compare
echo "=== Terminal Bench Results ==="
echo "Before:" 
ls -la results/before/
echo "After:"
ls -la results/after/