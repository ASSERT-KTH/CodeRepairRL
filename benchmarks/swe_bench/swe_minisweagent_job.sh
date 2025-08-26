#!/bin/bash
set -euo pipefail
# SWE-bench evaluation using minisweagent.
# Assumes vLLM is running on http://localhost:8000/v1 (see benchmarks/vllm.sh).

mkdir -p swe_bench/results_mini

echo "Running baseline evaluation..."
apptainer exec benchmarks/benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset verified --split dev --slice :25 \
  -m openai/gpt-4o \
  -w 4 \
  -o swe_bench/results_mini/before

echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
apptainer exec benchmarks/benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset verified --split dev --slice :25 \
  -m openai/nano \
  -w 4 \
  -o swe_bench/results_mini/after

echo "=== Scoring with SWE Bench evaluator ==="
apptainer exec benchmarks/benchmark_container.sif \
  sb-cli submit swe-bench-v dev \
  --predictions_path swe_bench/results_mini/before/preds.json \
  --output_dir swe_bench/results_mini/before_report

apptainer exec benchmarks/benchmark_container.sif \
  sb-cli submit swe-bench-v dev \
  --predictions_path swe_bench/results_mini/after/preds.json \
  --output_dir swe_bench/results_mini/after_report

echo "=== SWE Bench Results ==="
echo "Before report:"
cat swe_bench/results_mini/before_report/report.json | grep "resolved" || echo "No baseline results"
echo "After report:"
cat swe_bench/results_mini/after_report/report.json | grep "resolved" || echo "No LoRA results"