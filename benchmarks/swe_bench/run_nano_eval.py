#!/usr/bin/env python3
"""
Run SWE-bench evaluation using nano_agent.py
"""

import json
import sys
import argparse
from pathlib import Path

# Add parent dir to path to import nano_agent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.nano_agent import _process_one, NanoConfig
from datasets import load_dataset


def run_evaluation(endpoint: str, model_name: str, subset: str, split: str, slice_spec: str, output_dir: Path):
    """Run nano_agent on SWE-bench tasks and save predictions."""
    
    # Load SWE-bench dataset
    dataset = load_dataset(f"princeton-nlp/SWE-bench_{subset}", split=split)
    
    # Parse slice (e.g., ":25" means first 25)
    if slice_spec.startswith(":"):
        dataset = dataset.select(range(int(slice_spec[1:])))
    
    # Setup config for nano_agent
    config = NanoConfig(
        base_url=endpoint,
        model=model_name,  # Use the provided model name (e.g., "nano" for LoRA)
        timeout=300,
        temperature=0.0
    )
    
    predictions = {}
    
    for idx, instance in enumerate(dataset):
        print(f"Processing {idx+1}/{len(dataset)}: {instance['instance_id']}")
        
        # Prepare data for nano_agent
        data = {
            "instance_id": instance["instance_id"],
            "problem_statement": instance["problem_statement"],
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": instance.get("version", ""),
            # Add any other fields nano_agent expects
        }
        
        try:
            # Run nano_agent
            result = _process_one(data, config)
            
            # Extract the patch/diff from result
            # Assuming nano_agent returns a dict with 'patch' or 'diff' key
            patch = result.get("patch", "") or result.get("diff", "") or result.get("model_patch", "")
            
            if patch:
                # Format for SWE-bench submission (dict format)
                predictions[instance["instance_id"]] = {
                    "model_patch": patch,
                    "model_name_or_path": f"nano-agent-{config.model}"
                }
                
        except Exception as e:
            print(f"Error processing {instance['instance_id']}: {e}")
            # Still add empty prediction to track attempted instances
            predictions[instance["instance_id"]] = {
                "model_patch": "",
                "model_name_or_path": f"nano-agent-{config.model}"
            }
    
    # Save predictions in SWE-bench format
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_file = output_dir / "preds.json"
    
    with open(preds_file, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"Saved predictions to {preds_file}")
    
    # Also save in JSONL format for local evaluation
    jsonl_file = output_dir / "preds.jsonl"
    with open(jsonl_file, "w") as f:
        for instance_id, pred_data in predictions.items():
            obj = {
                "instance_id": instance_id,
                "model": pred_data["model_name_or_path"],
                "prediction": pred_data["model_patch"]
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
    parser.add_argument("--split", default="dev",
                        help="Dataset split")
    parser.add_argument("--slice", default=":25",
                        help="Slice of dataset to run (e.g., :25 for first 25)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    run_evaluation(args.endpoint, args.model_name, args.subset, args.split, args.slice, output_dir)


if __name__ == "__main__":
    main()