#!/bin/bash
set -euo pipefail
# SWE-bench evaluation using nano_agent.
# Assumes vLLM is running on http://localhost:8000/v1 (see benchmarks/vllm.sh).

mkdir -p swe_bench/results_nano
echo "Running nano_agent evaluation..."
apptainer exec benchmarks/benchmark_container.sif python benchmarks/swe_bench/run_nano_eval.py \
  --endpoint "http://localhost:8000/v1" \
  --model-name "nano" \
  --output-dir swe_bench/results_nano \
  --subset verified \
  --split dev \
  --slice :25

echo "=== Scoring with SWE Bench evaluator ==="
apptainer exec benchmarks/benchmark_container.sif \
  sb-cli submit swe-bench-v dev \
  --predictions_path swe_bench/results_nano/preds.json \
  --output_dir swe_bench/results_nano/report

echo "=== SWE Bench Results ==="
cat swe_bench/results_nano/report/report.json | grep "resolved" || echo "No results found"