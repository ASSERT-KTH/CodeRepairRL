import logging
import hashlib
from typing import Any

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def _get_swe_gym_split(dataset_name: str, holdout_partition: bool, holdout_ratio: float = 0.25) -> Dataset:
    """
    Internal function to load and split the SWE-Gym dataset.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_partition: If True, return holdout partition; if False, return repo repair partition
        holdout_ratio: Ratio of data to allocate to holdout partition (default 0.5)
        
    Returns:
        The requested partition of the dataset
    """
    logger.info(f"Loading SWE-bench dataset: {dataset_name}")
    
    # Load the SWE-bench dataset
    swe_ds = load_dataset(dataset_name)
    swe_ds = swe_ds.get("train") or swe_ds.get("test")
    
    # Create deterministic split based on instance_id hash
    def should_be_holdout(example):
        # Use MD5 hash of instance_id for deterministic splitting
        hash_val = int(hashlib.md5(example['instance_id'].encode()).hexdigest(), 16)
        # Convert to [0, 1] range and compare with holdout_ratio
        return (hash_val / (16**32)) < holdout_ratio
    
    # Filter based on partition type
    if holdout_partition:
        swe_ds = swe_ds.filter(should_be_holdout)
        logger.info(f"Creating holdout dataset with {len(swe_ds)} examples")
    else:
        swe_ds = swe_ds.filter(lambda x: not should_be_holdout(x))
        logger.info(f"Creating repository repair dataset with {len(swe_ds)} examples")
    
    # Add a dummy "prompt" key for compatibility with trl
    swe_ds = swe_ds.map(lambda x: {"prompt": "Dummy"})
    
    return swe_ds

# mirroring the other data methods though not strictly doing much
def get_swe_gym_repo_repair_dataset(
    dataset_name: str,
    holdout_ratio: float = 0.0,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset and convert it to a repository repair dataset.
    This function returns the non-holdout partition of the data for RL/GRPO training.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_ratio: Ratio of data to allocate to holdout partition (default 0.5)
        
    Returns:
        The processed dataset (repo repair partition)
    """
    return _get_swe_gym_split(dataset_name, holdout_partition=False, holdout_ratio=holdout_ratio)

def get_swe_gym_holdout_dataset(
    dataset_name: str,
    holdout_ratio: float = 0.25,
    **kwargs  # absorbs additional arguments required by the other get functions
) -> Dataset:
    """
    Load the SWE-bench dataset for SFT data holdout via rejection sampling.
    This function returns the holdout partition of the data, ensuring no overlap 
    with the repository repair dataset. Used by curate_sft_data.py to generate
    high-quality SFT examples through multiple rollouts and filtering.
    
    Args:
        dataset_name: HuggingFace dataset name for SWE-bench
        holdout_ratio: Ratio of data to allocate to holdout partition (default 0.5)
        
    Returns:
        The processed dataset (holdout partition for rejection sampling)
    """
    return _get_swe_gym_split(dataset_name, holdout_partition=True, holdout_ratio=holdout_ratio)


def get_swe_gym_formatted_sft_dataset(
    dataset_name: str,
    reward_min: float = 0.2,
    **kwargs
) -> Dataset:
    """
    Load and format a curated SFT dataset for training.
    This function loads an already-curated dataset (created by curate_sft_data.py)
    and formats it for SFT training.
    
    Args:
        dataset_name: HuggingFace dataset name for curated SFT data
        reward_min: Minimum reward for rejection sampling
        
    Returns:
        The formatted dataset ready for SFT training
    """
    logger.info(f"Loading curated SFT dataset: {dataset_name}")
    
    # Load the curated dataset
    dataset = load_dataset(dataset_name, split="train")
    
    logger.info(f"Preparing dataset with {len(dataset)} examples...")
    original_size = len(dataset)
    
    dataset = dataset.filter(lambda x: x["reward"] > reward_min)
    logger.info(f"Filtered dataset from {original_size} to {len(dataset)} examples")
    
    return dataset

if __name__ == "__main__":
    ds = load_dataset("SWE-Gym/SWE-Gym-Lite")
    print(ds)
    
    # Test the split functions
    holdout_ds = get_swe_gym_holdout_dataset()
    repair_ds = get_swe_gym_repo_repair_dataset()
    
    print(f"Holdout dataset size: {len(holdout_ds)}")
    print(f"Repo repair dataset size: {len(repair_ds)}")
    print(f"Total: {len(holdout_ds) + len(repair_ds)}")
    
    # Verify no overlap
    holdout_ids = set(holdout_ds['instance_id'])
    repair_ids = set(repair_ds['instance_id'])
    overlap = holdout_ids.intersection(repair_ids)
    print(f"Overlap between partitions: {len(overlap)} items")