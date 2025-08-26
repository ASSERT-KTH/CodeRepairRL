#!/bin/bash
set -euo pipefail
# Tau Bench evaluation.
# Assumes vLLM is running on http://localhost:8000/v1 (see benchmarks/vllm.sh).

mkdir -p tau_bench/results

echo "Running baseline evaluation..."
apptainer exec benchmarks/benchmark_container.sif python -m tau_bench.run \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 4 \
  --output-dir tau_bench/results/before

echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
apptainer exec benchmarks/benchmark_container.sif python -m tau_bench.run \
  --agent-strategy tool-calling \
  --env retail \
  --model nano --model-provider openai \
  --user-model gpt-4o --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 4 \
  --output-dir tau_bench/results/after

echo "=== Tau Bench Results ==="
echo "Before:"
cat tau_bench/results/before/metrics.json | grep success_rate || echo "No baseline metrics found"
echo "After:"
cat tau_bench/results/after/metrics.json | grep success_rate || echo "No LoRA metrics found"