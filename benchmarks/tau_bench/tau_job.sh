#!/bin/bash
#SBATCH --job-name=tau-bench-eval
#SBATCH --output=logs/tau_bench_%j.out
#SBATCH --error=logs/tau_bench_%j.err
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --time=4:00:00
#SBATCH -C "fat"

# Tau Bench evaluation with vLLM server

LORA_PATH=${1:-"/proj/berzelius-2024-336/users/x_bjabj/models/nano_lora"}
BASE_MODEL="Qwen/Qwen3-14B"
PORT=8001  # Unique port for tau_bench

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
cd /workspace/benchmarks/tau_bench
mkdir -p results

# Before (baseline)
echo "Running baseline evaluation..."
apptainer exec benchmark_container.sif python -m tau_bench.run \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 4 \
  --output-dir results/before

# After (LoRA model)
echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="dummy"

apptainer exec benchmark_container.sif python -m tau_bench.run \
  --agent-strategy tool-calling \
  --env retail \
  --model nano --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 4 \
  --output-dir results/after

# Kill vLLM server
kill $VLLM_PID

# Simple comparison
echo "=== Tau Bench Results ==="
echo "Before:"
cat results/before/metrics.json | grep success_rate || echo "No baseline metrics found"
echo "After:"  
cat results/after/metrics.json | grep success_rate || echo "No LoRA metrics found"