#!/usr/bin/env python3
"""
Merge preds.jsonl files from multiple shards into a single file.

Defaults assume the job array wrote shard outputs to:
  benchmarks/swe_bench/results_nano/shard_<TASK_ID>/preds.jsonl

Usage examples:
  - Merge all shard_* under results_nano to preds.jsonl in the same root:
      python3 benchmarks/swe_bench/merge_preds.py \
        --input-root benchmarks/swe_bench/results_nano \
        --output benchmarks/swe_bench/results_nano/preds.jsonl

  - Merge explicit files:
      python3 benchmarks/swe_bench/merge_preds.py \
        --inputs shard_0/preds.jsonl shard_1/preds.jsonl \
        --output benchmarks/swe_bench/results_nano/preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge shard preds.jsonl files into one.")
    parser.add_argument(
        "--input-root",
        default="benchmarks/swe_bench/results_nano",
        help="Root directory containing shard_* subdirectories",
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
        default="benchmarks/swe_bench/results_nano/preds.jsonl",
        help="Path to write merged preds.jsonl",
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


def discover_input_files(root: Path, pattern: str) -> List[Path]:
    shard_dirs = sorted((p for p in root.glob(pattern) if p.is_dir()), key=numeric_suffix)
    inputs: List[Path] = []
    for shard_dir in shard_dirs:
        candidate = shard_dir / "preds.jsonl"
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
        input_files = discover_input_files(input_root, args.pattern)

    if not input_files:
        print("[error] No input files found", file=sys.stderr)
        sys.exit(2)

    print(f"Merging {len(input_files)} file(s):")
    for p in input_files:
        print(f" - {p}")

    merged = merge_preds(input_files, args.dedup)
    print(f"Total merged instances: {len(merged)}")

    output_path = Path(args.output)
    write_jsonl(merged, output_path)
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()


