#!/bin/bash
set -euo pipefail
# SWE-bench evaluation using minisweagent.
# Assumes vLLM is running on http://localhost:8000/v1 (see benchmarks/vllm.sh).

mkdir -p swe_bench/results_mini

BASELINE_MODEL="openai/gpt-4o"
LORA_MODEL="openai/nano"
SUBSET="verified"
SPLIT="dev"
SLICE=":25"
WORKERS=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-model)
      BASELINE_MODEL="$2"; shift 2;;
    --lora-model)
      LORA_MODEL="$2"; shift 2;;
    --subset)
      SUBSET="$2"; shift 2;;
    --split)
      SPLIT="$2"; shift 2;;
    --slice)
      SLICE="$2"; shift 2;;
    --workers)
      WORKERS="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "Running baseline evaluation..."
apptainer exec benchmarks/benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset "$SUBSET" --split "$SPLIT" --slice "$SLICE" \
  -m "$BASELINE_MODEL" \
  -w "$WORKERS" \
  -o swe_bench/results_mini/before

echo "Running LoRA model evaluation..."
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="dummy"
apptainer exec benchmarks/benchmark_container.sif python -m minisweagent.run.extra.swebench \
  --subset "$SUBSET" --split "$SPLIT" --slice "$SLICE" \
  -m "$LORA_MODEL" \
  -w "$WORKERS" \
  -o swe_bench/results_mini/after

echo "=== Scoring with SWE Bench evaluator ==="
echo "Harness should be run on a CPU server. Use the generated JSONL files:"
echo "  before: $(pwd)/swe_bench/results_mini/before/preds.jsonl"
echo "  after:  $(pwd)/swe_bench/results_mini/after/preds.jsonl"
echo "Example harness invocation:" 
echo "  benchmarks/swe_bench/run_harness_eval.sh --subset $SUBSET --split $SPLIT \\\n    --preds $(pwd)/swe_bench/results_mini/before/preds.jsonl --run-id $(basename \"$BASELINE_MODEL\")_${SPLIT}"
echo "  benchmarks/swe_bench/run_harness_eval.sh --subset $SUBSET --split $SPLIT \\\n    --preds $(pwd)/swe_bench/results_mini/after/preds.jsonl --run-id $(basename \"$LORA_MODEL\")_${SPLIT}"