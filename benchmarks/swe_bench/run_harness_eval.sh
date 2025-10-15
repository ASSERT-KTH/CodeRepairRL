#!/usr/bin/env bash
set -euo pipefail

# Run SWE-bench harness locally on a CPU server.
# Usage:
#   benchmarks/swe_bench/run_harness_eval.sh \
#     --subset verified --split test \
#     --preds /abs/path/to/preds.jsonl \
#     --run-id my_run \
#     [--max-workers 8]
#
# Requirements (on this CPU server):
#   pip install swebench
#   Docker installed and running

subset="verified"
split="test"
preds=""
run_id="swebench_local_run"
max_workers="8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subset)
      subset="$2"; shift 2;;
    --split)
      split="$2"; shift 2;;
    --preds)
      preds="$2"; shift 2;;
    --run-id)
      run_id="$2"; shift 2;;
    --max-workers)
      max_workers="$2"; shift 2;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$preds" ]]; then
  echo "ERROR: --preds /path/to/preds.jsonl is required" >&2
  exit 1
fi

case "$subset" in
  verified|Verified)
    dataset_name="princeton-nlp/SWE-bench_Verified";;
  lite|Lite)
    dataset_name="princeton-nlp/SWE-bench_Lite";;
  full|Full|main)
    dataset_name="princeton-nlp/SWE-bench";;
  *)
    echo "ERROR: Unknown subset '$subset' (expected: verified|lite|full)" >&2
    exit 1;;
esac

echo "Running SWE-bench harness..."
echo "  dataset_name: $dataset_name"
echo "  split:        $split"
echo "  predictions:  $preds"
echo "  run_id:       $run_id"
echo "  max_workers:  $max_workers"

python -m swebench.harness.run_evaluation \
  --dataset_name "$dataset_name" \
  --split "$split" \
  --predictions_path "$preds" \
  --max_workers "$max_workers" \
  --run_id "$run_id" \
  --cache_level "instance" \
  --timeout 3600


