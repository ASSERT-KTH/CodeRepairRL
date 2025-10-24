#!/usr/bin/env python3
"""
Run SWE-bench evaluation using mini_agent.py (Mini-SWE-Agent backend)

Interface mirrors run_nano_eval.py to keep flows consistent across agents.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path to import mini_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.mini_agent import _process_one, AgentConfig  # type: ignore
from datasets import load_dataset


def run_evaluation(endpoint: str, model_name: str, subset: str, split: str, slice_spec: str, output_dir: Path):
    """Run mini_agent on SWE-bench tasks and save predictions using a process pool."""

    # Load SWE-bench dataset
    dataset = load_dataset(f"princeton-nlp/SWE-bench_{subset}", split=split)

    # Parse slice
    # Supported forms:
    #   ":N"        -> first N instances
    #   "start:end" -> instances in [start, end) zero-based half-open interval
    if slice_spec:
        if slice_spec.startswith(":"):
            dataset = dataset.select(range(int(slice_spec[1:])))
        elif ":" in slice_spec:
            start_str, end_str = slice_spec.split(":", 1)
            start_idx = int(start_str)
            end_idx = int(end_str)
            if start_idx < 0 or end_idx < 0:
                raise ValueError("slice must be non-negative indices")
            if end_idx < start_idx:
                raise ValueError("slice end must be >= start")
            dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))

    # Setup config for mini_agent (uses Litellm/OpenAI-compatible endpoint)
    config = AgentConfig(
        api_base=endpoint,
        model=model_name,  # e.g., "hosted_vllm/Qwen/Qwen3-8B"
        token_limit=16384,
        time_limit=40,
        tool_limit=30,
        temperature=0.2,
    )

    # Prepare inputs for workers
    inputs: list[dict] = []
    for instance in dataset:
        inputs.append({
            "instance_id": instance["instance_id"],
            "problem_statement": instance["problem_statement"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": instance.get("version", ""),
        })

    predictions: dict[str, dict] = {}
    detailed_predictions: dict[str, dict] = {}

    # Run with a process pool of up to 8 workers
    max_workers = min(8, len(inputs)) if inputs else 0
    if max_workers == 0:
        print("No instances to process.")
        return

    print(f"Starting processing {len(inputs)} instances with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance_id = {
            executor.submit(_process_one, datum, config): datum["instance_id"] for datum in inputs
        }

        completed = 0
        for future in as_completed(future_to_instance_id):
            instance_id = future_to_instance_id[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing {instance_id}: {e}")
                result = {}

            # Extract model patch; prefer 'generated_diff' produced by mini_agent
            patch = (
                result.get("generated_diff", "")
                or result.get("patch", "")
                or result.get("diff", "")
                or result.get("model_patch", "")
            )

            predictions[instance_id] = {
                "model_patch": patch or "",
                "model_name_or_path": f"mini-agent-{config.model}",
            }

            if result:
                detailed_predictions[instance_id] = result

            completed += 1
            if completed % 5 == 0 or completed == len(inputs):
                print(f"Progress: {completed}/{len(inputs)} completed")
    
    # Save predictions in JSONL format for SWE-bench harness
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / "preds.jsonl"
    with open(jsonl_file, "w") as f:
        for instance_id, pred_data in predictions.items():
            obj = {
                "instance_id": instance_id,
                "model_name_or_path": pred_data["model_name_or_path"],
                "model_patch": pred_data["model_patch"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Saved JSONL format to {jsonl_file}")

    # Save detailed predictions (entire result dictionaries)
    detailed_file = output_dir / "detailed_predictions.jsonl"
    with open(detailed_file, "w") as f:
        for instance_id, det in detailed_predictions.items():
            obj = {"instance_id": instance_id, "detailed_predictions": det}
            f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    print(f"Saved detailed predictions to {detailed_file}")
    
    # Quick validation - check if patches can apply
    valid_count = 0
    for instance_id, pred_data in predictions.items():
        if pred_data["model_patch"]:
            valid_count += 1
    
    print(f"\nSummary: {valid_count}/{len(predictions)} instances have non-empty patches")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench eval with mini_agent")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="Model endpoint URL")
    parser.add_argument("--model-name", default="hosted_vllm/Qwen/Qwen3-8B",
                        help="Model name passed to mini agent")
    parser.add_argument("--output-dir", default="swe_bench/results_mini",
                        help="Output directory for results")
    parser.add_argument("--subset", default="verified",
                        help="SWE-bench subset (verified, lite, full)")
    parser.add_argument("--split", default="test",
                        help="Dataset split")
    parser.add_argument("--slice", default=":25",
                        help="Slice to run. Forms: :N (first N) or start:end (half-open)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_evaluation(args.endpoint, args.model_name, args.subset, args.split, args.slice, output_dir)


if __name__ == "__main__":
    main()


