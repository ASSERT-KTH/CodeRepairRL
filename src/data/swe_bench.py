import logging
from typing import Literal

from datasets import load_dataset, Dataset

from src.utils.git import handle_to_url
from src.utils.diff import SearchReplaceDiff
from src.data.code_repo_repair import create_repo_repair_dataset

logger = logging.getLogger(__name__)

# Cloning the repos must happen on the trl side (in our custom fork)
def get_swe_bench_dataset(
    split: Literal["train", "dev"] = "dev",
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",  # or any of the other SWE-bench datasets
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    
    Args:
        tokenizer: Tokenizer (passed for consistency with other dataset loaders)
        split: Dataset split to use ("train" or "dev")
        dataset_name: HuggingFace dataset name for SWE-bench
        
    Returns:
        The processed dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}, split: {split}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)[split]
    
    logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")
    
    return create_repo_repair_dataset(
        repo_urls=[handle_to_url(item["repo"]) for item in swe_ds],
        repo_commit_hashes=[item["base_commit"] for item in swe_ds],
        search_replace_patches=[SearchReplaceDiff.from_unified_diff(item["patch"]).to_string() for item in swe_ds],
        descriptions=[item["problem_statement"] for item in swe_ds],
    )




