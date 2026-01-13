#!/usr/bin/env bash
set -euo pipefail

# Run SWE-bench harness evaluation for all available prediction files.
# Usage:
#   benchmarks/swe_bench/run_all_harness_evals.sh [--max-workers 8] [--dry-run]
#
# This script finds all preds.jsonl files in swe_bench/ and runs the harness
# evaluation for each one sequentially.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PREDS_DIR="$PROJECT_ROOT/swe_bench"
LOGS_DIR="$PROJECT_ROOT/eval_logs"

max_workers="8"
dry_run=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-workers)
      max_workers="$2"; shift 2;;
    --dry-run)
      dry_run=true; shift;;
    *)
      echo "Unknown arg: $1"; exit 1;;
  esac
done

# Create logs directory
mkdir -p "$LOGS_DIR"

# Find all preds.jsonl files
mapfile -t pred_files < <(find "$PREDS_DIR" -name "preds.jsonl" -type f | sort)

if [[ ${#pred_files[@]} -eq 0 ]]; then
  echo "No preds.jsonl files found in $PREDS_DIR"
  exit 1
fi

echo "Found ${#pred_files[@]} prediction files to evaluate"
echo "Max workers per job: $max_workers"
echo "Logs directory: $LOGS_DIR"
echo ""

for pred_file in "${pred_files[@]}"; do
  # Extract run ID from path: swe_bench/<model_name>/run_N/preds.jsonl
  # -> run_id: <model_name>__run_N__<random_suffix>
  # Random suffix avoids caching issues with the eval harness
  rel_path="${pred_file#$PREDS_DIR/}"
  model_name=$(echo "$rel_path" | cut -d'/' -f1)
  run_name=$(echo "$rel_path" | cut -d'/' -f2)
  random_suffix=$(head -c 100 /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
  run_id="${model_name}__${run_name}__${random_suffix}"
  
  log_file="$LOGS_DIR/${run_id}.out"
  
  echo "Run: $run_id"
  echo "  Predictions: $pred_file"
  echo "  Log file: $log_file"
  
  if [[ "$dry_run" == true ]]; then
    echo "  [DRY RUN] Would execute:"
    echo "    $SCRIPT_DIR/run_harness_eval.sh --preds $pred_file --run-id $run_id --max-workers $max_workers 2>&1 | tee $log_file"
  else
    echo "  Running..."
    "$SCRIPT_DIR/run_harness_eval.sh" \
      --preds "$pred_file" \
      --run-id "$run_id" \
      --max-workers "$max_workers" \
      2>&1 | tee "$log_file"
    echo "  Completed."
  fi
  echo ""
done

echo "All evaluations completed."
