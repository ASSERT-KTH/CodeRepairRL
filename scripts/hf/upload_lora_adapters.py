#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload selected LoRA adapter checkpoints to Hugging Face"
    )

    default_source = \
        "/proj/berzelius-2024-336/users/x_bjabj/CodeRepairRL/outputs/Qwen3-14B-GSPO-SWE-Gym-Full"

    parser.add_argument(
        "--source-dir",
        type=str,
        default=default_source,
        help=f"Directory containing checkpoint-* folders (default: {default_source})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        required=True,
        help="One or more checkpoint step numbers to upload (e.g. 200 220 240)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=None,
        help="Hugging Face org (or username). If omitted, inferred from token",
    )
    parser.add_argument(
        "--repo-template",
        type=str,
        default="{org}/{base_name}-lora-step-{step}",
        help=(
            "Template for repo_id. Placeholders: {org}, {user}, {base_name}, {step}. "
            "Example: '{org}/Qwen3-14B-GSPO-SWE-Gym-Full-lora-{step}'"
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repositories as private",
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        nargs="+",
        default=[
            "adapter_model.safetensors",
            "adapter_config.json",
            "README.md",
        ],
        help=(
            "Whitelist of file patterns to upload inside each checkpoint folder. "
            "Defaults to adapter files and README."
        ),
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload LoRA adapter: step {step}",
        help="Commit message template. Placeholders: {step}",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                or os.environ.get("HF_API_TOKEN"),
        help="Hugging Face token. Defaults to env HF_TOKEN/HUGGINGFACE_HUB_TOKEN",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without creating repos or uploading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def infer_identity(api: HfApi, provided_org: Optional[str]) -> str:
    if provided_org:
        return provided_org
    try:
        who = api.whoami()
        return who.get("orgs")[0] if who.get("orgs") else who.get("name")
    except Exception:
        raise RuntimeError(
            "Unable to infer org/user. Provide --org or ensure token is valid."
        )


def build_repo_id(template: str, org_or_user: str, base_name: str, step: int) -> str:
    # The template may use {org} or {user}; map both to the same value for convenience
    return template.format(org=org_or_user, user=org_or_user, base_name=base_name, step=step)


def find_checkpoint_dir(source_dir: Path, step: int) -> Path:
    path = source_dir / f"checkpoint-{step}"
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {path}")
    return path


def validate_is_lora_checkpoint(checkpoint_dir: Path) -> None:
    required = ["adapter_model.safetensors", "adapter_config.json"]
    for fname in required:
        if not (checkpoint_dir / fname).exists():
            raise FileNotFoundError(f"Missing required file in {checkpoint_dir}: {fname}")


def upload_one_checkpoint(
    api: HfApi,
    checkpoint_dir: Path,
    repo_id: str,
    allow_patterns: List[str],
    private: bool,
    commit_message: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    if verbose:
        print(f"Repo: {repo_id}")
        print(f"  From: {checkpoint_dir}")
        print(f"  Allow patterns: {allow_patterns}")
        print(f"  Private: {private}")
        print(f"  Dry run: {dry_run}")

    if dry_run:
        # Show matched files
        import fnmatch

        matched = set()
        for pattern in allow_patterns:
            for f in checkpoint_dir.rglob("*"):
                if f.is_file():
                    rel = f.relative_to(checkpoint_dir).as_posix()
                    if fnmatch.fnmatch(rel, pattern):
                        matched.add(rel)
        for rel in sorted(matched):
            print(f"  Would upload: {rel}")
        return

    # Ensure repo exists
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # Upload selected files from the checkpoint directory root
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint_dir),
        allow_patterns=allow_patterns,
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        print(f"Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(2)

    base_name = source_dir.name
    api = HfApi(token=args.token)
    org_or_user = infer_identity(api, args.org)

    steps_sorted = sorted(set(args.steps))
    for step in steps_sorted:
        try:
            checkpoint_dir = find_checkpoint_dir(source_dir, step)
            validate_is_lora_checkpoint(checkpoint_dir)
            repo_id = build_repo_id(args.repo_template, org_or_user, base_name, step)
            commit_message = args.commit_message.format(step=step)
            upload_one_checkpoint(
                api=api,
                checkpoint_dir=checkpoint_dir,
                repo_id=repo_id,
                allow_patterns=args.allow_patterns,
                private=args.private,
                commit_message=commit_message,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error for step {step}: {e}", file=sys.stderr)
            # Continue with other steps


if __name__ == "__main__":
    main()



