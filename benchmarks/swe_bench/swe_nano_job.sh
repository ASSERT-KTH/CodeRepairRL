#!/bin/bash
set -euo pipefail
# SWE-bench evaluation using nano_agent.
# Assumes vLLM is running on http://localhost:8000/v1 (see benchmarks/vllm.sh).

MODEL_NAME="nano"
ENDPOINT="http://localhost:8000/v1"
OUTPUT_DIR="swe_bench/results_nano"
SUBSET="verified"
SPLIT="test"
SLICE=":25"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name)
      MODEL_NAME="$2"; shift 2;;
    --endpoint)
      ENDPOINT="$2"; shift 2;;
    --output-dir)
      OUTPUT_DIR="$2"; shift 2;;
    --subset)
      SUBSET="$2"; shift 2;;
    --split)
      SPLIT="$2"; shift 2;;
    --slice)
      SLICE="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUTPUT_DIR"

echo "Running nano_agent evaluation with model '$MODEL_NAME'..."
apptainer exec benchmarks/benchmark_container.sif python3 benchmarks/swe_bench/run_nano_eval.py \
  --endpoint "$ENDPOINT" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --slice "$SLICE"

echo "Predictions saved to $OUTPUT_DIR/preds.jsonl"
echo "Now run the SWE-bench harness on a CPU server, e.g.:"
echo "  benchmarks/swe_bench/run_harness_eval.sh --subset $SUBSET --split $SPLIT \\\n    --preds $(pwd)/$OUTPUT_DIR/preds.jsonl --run-id ${MODEL_NAME}_${SPLIT}"
