"""Loader for SWE-bench instances from HuggingFace datasets."""

import json
import logging
from pathlib import Path
from typing import Any

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from ..models.instance import Instance

logger = logging.getLogger(__name__)


class SWEBenchInstanceLoader:
    """Loader for SWE-bench instances from HuggingFace datasets or local JSON/JSONL files."""
    
    def load_instances(
        self,
        source: str | Path,
        split: str | None = None,
    ) -> dict[str, Instance]:
        """Load instances from a HuggingFace dataset or local file.
        
        Args:
            source: HuggingFace dataset name (e.g., 'SWE-bench/SWE-bench') or path to JSON/JSONL file
            split: Dataset split to load (e.g., 'test', 'train'). If None, uses default split.
            
        Returns:
            Dictionary mapping instance_id to Instance objects
        """
        source_str = str(source)
        
        # Check if it's a file path
        if Path(source_str).exists():
            return self._load_from_file(Path(source_str))
        
        # Otherwise, try loading from HuggingFace
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets library not available. Install with: pip install datasets"
            )
        
        return self._load_from_huggingface(source_str, split)
    
    def _load_from_huggingface(
        self,
        dataset_name: str,
        split: str | None = None,
    ) -> dict[str, Instance]:
        """Load instances from HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to load
            
        Returns:
            Dictionary mapping instance_id to Instance objects
        """
        logger.info(f"Loading SWE-bench instances from HuggingFace: {dataset_name}")
        
        try:
            if split:
                dataset = load_dataset(dataset_name, split=split)
            else:
                # Try to load default split
                dataset_dict = load_dataset(dataset_name)
                # Prefer 'test' split, then 'train', then first available
                if 'test' in dataset_dict:
                    dataset = dataset_dict['test']
                elif 'train' in dataset_dict:
                    dataset = dataset_dict['train']
                else:
                    dataset = dataset_dict[list(dataset_dict.keys())[0]]
            
            instances = {}
            for example in dataset:
                instance_id = example.get('instance_id')
                if not instance_id:
                    logger.warning(f"Skipping example without instance_id: {example.keys()}")
                    continue
                
                patch = example.get('patch', '')
                test_patch = example.get('test_patch') or example.get('test_patch_diff')
                
                instance = Instance(
                    instance_id=instance_id,
                    patch=patch,
                    test_patch=test_patch,
                )
                
                instances[instance_id] = instance
            
            logger.info(f"Loaded {len(instances)} instances from {dataset_name}")
            return instances
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def _load_from_file(self, file_path: Path) -> dict[str, Instance]:
        """Load instances from a local JSON or JSONL file.
        
        Args:
            file_path: Path to JSON or JSONL file
            
        Returns:
            Dictionary mapping instance_id to Instance objects
        """
        logger.info(f"Loading SWE-bench instances from file: {file_path}")
        
        instances = {}
        
        if file_path.suffix == '.jsonl':
            # JSONL format: one JSON object per line
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        instance = self._parse_instance(data)
                        if instance:
                            instances[instance.instance_id] = instance
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
        else:
            # JSON format: array of objects or single object
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                # Array of instances
                for item in data:
                    instance = self._parse_instance(item)
                    if instance:
                        instances[instance.instance_id] = instance
            elif isinstance(data, dict):
                # Single instance or dict mapping instance_id to instance data
                if 'instance_id' in data:
                    # Single instance
                    instance = self._parse_instance(data)
                    if instance:
                        instances[instance.instance_id] = instance
                else:
                    # Dict mapping instance_id to instance data
                    for instance_id, instance_data in data.items():
                        if isinstance(instance_data, dict):
                            instance_data['instance_id'] = instance_id
                        instance = self._parse_instance(instance_data)
                        if instance:
                            instances[instance.instance_id] = instance
        
        logger.info(f"Loaded {len(instances)} instances from {file_path}")
        return instances
    
    def _parse_instance(self, data: dict[str, Any]) -> Instance | None:
        """Parse a single instance from dictionary data.
        
        Args:
            data: Dictionary containing instance data
            
        Returns:
            Instance object or None if parsing fails
        """
        instance_id = data.get('instance_id')
        if not instance_id:
            logger.warning(f"Skipping instance without instance_id: {data.keys()}")
            return None
        
        patch = data.get('patch', '')
        test_patch = data.get('test_patch') or data.get('test_patch_diff')
        
        # If difficulty metrics are pre-computed, use them
        instance = Instance(
            instance_id=instance_id,
            patch=patch,
            test_patch=test_patch,
            num_files_changed=data.get('num_files_changed', 0),
            num_lines_changed=data.get('num_lines_changed', 0),
            num_additions=data.get('num_additions', 0),
            num_deletions=data.get('num_deletions', 0),
        )
        
        return instance
    
    def save_instances(
        self,
        instances: dict[str, Instance],
        output_path: Path,
        format: str = 'json',
    ) -> None:
        """Save instances to a file.
        
        Args:
            instances: Dictionary mapping instance_id to Instance objects
            output_path: Path to output file
            format: Output format ('json' or 'jsonl')
        """
        logger.info(f"Saving {len(instances)} instances to {output_path}")
        
        if format == 'jsonl':
            with open(output_path, 'w') as f:
                for instance in instances.values():
                    f.write(json.dumps(instance.to_dict()) + '\n')
        else:
            # JSON format: array of instances
            data = [instance.to_dict() for instance in instances.values()]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved instances to {output_path}")

