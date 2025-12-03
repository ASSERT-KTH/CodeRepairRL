#!/usr/bin/env python3
"""
Script to pull all SWE-bench images using apptainer pull to populate the cache.
Usage: python3 scripts/pull_swe_images.py --subset verified --shard-id 0 --num-shards 10
"""

import argparse
import subprocess
import sys
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Pull SWE-bench images for Apptainer cache")
    parser.add_argument("--subset", default="verified", help="SWE-bench subset (verified, lite, full)")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--temp-sif", default="/proj/berzelius-2024-336/users/x_andaf/CodeRepairRL/temp.sif", help="Temporary SIF file path")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard ID for distributed pulling")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    args = parser.parse_args()

    print(f"Loading dataset princeton-nlp/SWE-bench_{args.subset} split {args.split}...")
    try:
        dataset = load_dataset(f"princeton-nlp/SWE-bench_{args.subset}", split=args.split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Sharding logic
    if args.num_shards > 1:
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_id)
        print(f"Processing shard {args.shard_id}/{args.num_shards} with {len(dataset)} instances.")
    else:
        print(f"Found {len(dataset)} instances.")
    
    temp_sif = args.temp_sif
    
    # Ensure directory for temp file exists
    os.makedirs(os.path.dirname(temp_sif), exist_ok=True)

    for i, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        # Image naming convention: replace double underscore with _1776_
        image_name_tag = instance_id.replace("__", "_1776_")
        image_uri = f"docker://docker.io/swebench/sweb.eval.x86_64.{image_name_tag}:latest"
        
        print(f"[{i+1}/{len(dataset)}] Pulling {image_uri}...")
        try:
            # Pull to a temporary file to populate the cache
            # We can overwrite the same file since we are running sequentially
            subprocess.run(["apptainer", "pull", "--force", temp_sif, image_uri], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to pull {image_uri}: {e}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
            
    # Cleanup
    if os.path.exists(temp_sif):
        try:
            os.remove(temp_sif)
        except OSError:
            pass

    print("Done.")

if __name__ == "__main__":
    main()
