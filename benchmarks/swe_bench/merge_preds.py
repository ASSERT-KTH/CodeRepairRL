#!/usr/bin/env python3
"""
Merge preds.jsonl and detailed_predictions.jsonl files from multiple shards into single files.

Expected layout (per run):
  benchmarks/swe_bench/results/<run_tag>/shard_<TASK_ID>/preds.jsonl
  benchmarks/swe_bench/results/<run_tag>/shard_<TASK_ID>/detailed_predictions.jsonl

Where <run_tag> encodes scaffold and model, e.g.:
  nano-agent-hosted_vllm__Qwen__Qwen3-8B
  nano-agent-Qwen__Qwen3-8B__lora__nano_lora

Usage examples:
  - Merge all shard_* for a specific run_tag and write to run root:
      python3 benchmarks/swe_bench/merge_preds.py \
        --input-root benchmarks/swe_bench/results \
        --run-tag nano-agent-hosted_vllm__Qwen__Qwen3-8B \
        --output benchmarks/swe_bench/results/nano-agent-hosted_vllm__Qwen__Qwen3-8B/preds.jsonl

  - Merge when input-root is already the run directory:
      python3 benchmarks/swe_bench/merge_preds.py \
        --input-root benchmarks/swe_bench/results/nano-agent-hosted_vllm__Qwen__Qwen3-8B \
        --output benchmarks/swe_bench/results/nano-agent-hosted_vllm__Qwen__Qwen3-8B/preds.jsonl

  - Merge explicit files:
      python3 benchmarks/swe_bench/merge_preds.py \
        --inputs path/to/shard_0/preds.jsonl path/to/shard_1/preds.jsonl \
        --output path/to/preds.jsonl

Note: This script merges both preds.jsonl and detailed_predictions.jsonl when available.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard preds.jsonl files into one.")
    parser.add_argument(
        "--input-root",
        default="benchmarks/swe_bench/results",
        help="Either the run directory (<results>/<run_tag>) or its parent results directory",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional run tag (scaffold-model tag). If provided, shards are searched under <input-root>/<run-tag>",
    )
    parser.add_argument(
        "--pattern",
        default="shard_*",
        help="Glob pattern (relative to input-root) to find shard directories",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="Explicit list of preds.jsonl files to merge; overrides root/pattern",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Path to write merged preds.jsonl. Defaults to <run_root>/preds.jsonl when discoverable",
    )
    parser.add_argument(
        "--dedup",
        choices=["first", "last"],
        default="first",
        help="When duplicate instance_id appears, keep first or last occurrence",
    )
    return parser.parse_args()


def numeric_suffix(path: Path) -> Tuple[int, str]:
    """Return (numeric_suffix, name) for sorting shard directories.

    Examples:
      shard_0 -> (0, 'shard_0')
      shard_10 -> (10, 'shard_10')
      other -> (inf, 'other') to put nonconforming at the end
    """
    name = path.name
    for sep in ("_", "-"):
        if sep in name:
            prefix, suffix = name.rsplit(sep, 1)
            if suffix.isdigit():
                return (int(suffix), name)
    return (10**9, name)


def discover_input_files(root: Path, pattern: str, filename: str = "preds.jsonl") -> List[Path]:
    shard_dirs = sorted((p for p in root.glob(pattern) if p.is_dir()), key=numeric_suffix)
    inputs: List[Path] = []
    for shard_dir in shard_dirs:
        candidate = shard_dir / filename
        if candidate.is_file():
            inputs.append(candidate)
        else:
            print(f"[warn] Missing file: {candidate}", file=sys.stderr)
    return inputs


def iter_jsonl_lines(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[warn] {path}:{line_no} invalid JSON: {exc}", file=sys.stderr)
                        continue
                    if "instance_id" not in obj:
                        print(f"[warn] {path}:{line_no} missing 'instance_id'", file=sys.stderr)
                        continue
                    yield obj
        except FileNotFoundError:
            print(f"[warn] Not found: {path}", file=sys.stderr)


def merge_preds(inputs: List[Path], dedup: str) -> list[dict]:
    seen: dict[str, dict] = {}
    order: list[str] = []

    for obj in iter_jsonl_lines(inputs):
        instance_id = obj.get("instance_id", "")
        if not instance_id:
            continue
        if instance_id in seen:
            if dedup == "last":
                seen[instance_id] = obj
            else:
                # keep first
                pass
        else:
            seen[instance_id] = obj
            order.append(instance_id)

    if dedup == "last":
        # order remains as first-seen to preserve shard ordering
        pass

    return [seen[iid] for iid in order]


def write_jsonl(objs: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.inputs:
        input_files = [Path(p) for p in args.inputs]
    else:
        input_root = Path(args.input_root)
        run_root: Path
        # If a run-tag is provided, search under input_root/run_tag
        if args.run_tag:
            run_root = input_root / args.run_tag
        else:
            # If input_root already looks like a run directory (contains shard_*), use it directly.
            has_shards = any(p.is_dir() and p.name.startswith("shard_") for p in input_root.iterdir()) if input_root.exists() else False
            run_root = input_root if has_shards else input_root
        input_files = discover_input_files(run_root, args.pattern, "preds.jsonl")

    if not input_files:
        print("[error] No input files found", file=sys.stderr)
        sys.exit(2)

    print(f"Merging {len(input_files)} preds.jsonl file(s):")
    for p in input_files:
        print(f" - {p}")

    merged = merge_preds(input_files, args.dedup)
    print(f"Total merged instances: {len(merged)}")

    # Derive default output path if not provided and run_root is known
    if args.output:
        output_path = Path(args.output)
    else:
        # If args.run_tag was provided, write into that directory; else if input_root looks like run dir, write there.
        if args.run_tag:
            output_path = Path(args.input_root) / args.run_tag / "preds.jsonl"
        else:
            output_path = Path(args.input_root) / "preds.jsonl"
    write_jsonl(merged, output_path)
    print(f"Wrote: {output_path}")

    # Also merge detailed_predictions.jsonl if available
    if not args.inputs:
        detailed_input_files = discover_input_files(run_root, args.pattern, "detailed_predictions.jsonl")

        if detailed_input_files:
            print(f"\nMerging {len(detailed_input_files)} detailed_predictions.jsonl file(s):")
            for p in detailed_input_files:
                print(f" - {p}")

            detailed_merged = merge_preds(detailed_input_files, args.dedup)
            print(f"Total merged detailed instances: {len(detailed_merged)}")

            # Determine output path for detailed predictions
            detailed_output_path = output_path.parent / "detailed_predictions.jsonl"
            write_jsonl(detailed_merged, detailed_output_path)
            print(f"Wrote: {detailed_output_path}")
        else:
            print("\n[info] No detailed_predictions.jsonl files found to merge")


if __name__ == "__main__":
    main()


