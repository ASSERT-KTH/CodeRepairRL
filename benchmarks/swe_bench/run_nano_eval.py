#!/usr/bin/env python3
"""
Run SWE-bench evaluation using nano_agent.py

This file intentionally mirrors the interface of run_aider_eval.py to keep
eval flows consistent across agents.
"""

import json
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent dir to path to import nano_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.nano_agent import _process_one, NanoConfig
from datasets import load_dataset


def run_evaluation(endpoint: str, model_name: str, subset: str, split: str, slice_spec: str, output_dir: Path):
    """Run nano_agent on SWE-bench tasks and save predictions using a process pool."""

    # Load SWE-bench dataset
    dataset = load_dataset(f"princeton-nlp/SWE-bench_{subset}", split=split)

    # Parse slice (e.g., ":25" means first 25)
    if slice_spec.startswith(":"):
        dataset = dataset.select(range(int(slice_spec[1:])))

    # Setup config for nano_agent
    config = NanoConfig(
        api_base=endpoint,
        model=model_name,  # e.g., "nano" for LoRA
        time_limit=300,
        temperature=0.0,
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

            # Extract model patch; prefer 'generated_diff' produced by nano_agent
            patch = (
                result.get("generated_diff", "")
                or result.get("patch", "")
                or result.get("diff", "")
                or result.get("model_patch", "")
            )

            predictions[instance_id] = {
                "model_patch": patch or "",
                "model_name_or_path": f"nano-agent-{config.model}",
            }

            completed += 1
            if completed % 5 == 0 or completed == len(inputs):
                print(f"Progress: {completed}/{len(inputs)} completed")
    
    # Save predictions in SWE-bench format
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_file = output_dir / "preds.json"
    
    with open(preds_file, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved predictions to {preds_file}")
    
    # Also save in JSONL format for local evaluation (SWE-bench harness schema)
    jsonl_file = output_dir / "preds.jsonl"
    with open(jsonl_file, "w") as f:
        for instance_id, pred_data in predictions.items():
            obj = {
                "instance_id": instance_id,
                "model_name_or_path": pred_data["model_name_or_path"],
                "model_patch": pred_data["model_patch"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    
    print(f"Also saved JSONL format to {jsonl_file}")
    
    # Quick validation - check if patches can apply
    valid_count = 0
    for instance_id, pred_data in predictions.items():
        if pred_data["model_patch"]:
            valid_count += 1
    
    print(f"\nSummary: {valid_count}/{len(predictions)} instances have non-empty patches")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench eval with nano_agent")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1",
                        help="Model endpoint URL")
    parser.add_argument("--model-name", default="nano",
                        help="Model name to use (e.g., 'nano' for LoRA, base model name for baseline)")
    parser.add_argument("--output-dir", default="swe_bench/results_nano",
                        help="Output directory for results")
    parser.add_argument("--subset", default="verified",
                        help="SWE-bench subset (verified, lite, full)")
    parser.add_argument("--split", default="test",
                        help="Dataset split")
    parser.add_argument("--slice", default=":25",
                        help="Slice of dataset to run (e.g., :25 for first 25)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_evaluation(args.endpoint, args.model_name, args.subset, args.split, args.slice, output_dir)


if __name__ == "__main__":
    main()
