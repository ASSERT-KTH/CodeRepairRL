"""Abstract base class for trajectory loaders."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..models import Run, Trajectory

logger = logging.getLogger(__name__)


class TrajectoryLoader(ABC):
    """Abstract base class for loading trajectories from different formats."""
    
    @abstractmethod
    def load_trajectories(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from a file or directory.
        
        Args:
            path: Path to trajectory file(s)
            
        Returns:
            List of Trajectory objects
        """
        pass
    
    def load_run(
        self,
        name: str,
        scaffold: str,
        base_model: str,
        trajectories_path: Path | str,
        results_path: Path | str | None = None,
        lora_adapter: str | None = None,
    ) -> Run:
        """Load a complete run with trajectories and optional results.
        
        Args:
            name: Run identifier
            scaffold: Agent scaffold type
            base_model: Base model identifier
            trajectories_path: Path to trajectory file(s)
            results_path: Optional path to SWE-bench results JSON
            lora_adapter: Optional LoRA adapter name
            
        Returns:
            Run object with loaded trajectories
        """
        trajectories = self.load_trajectories(trajectories_path)
        
        resolved_ids: set[str] = set()
        if results_path:
            resolved_ids = self.load_results(results_path)
        
        run = Run(
            name=name,
            scaffold=scaffold,
            base_model=base_model,
            trajectories=trajectories,
            resolved_ids=resolved_ids,
            lora_adapter=lora_adapter,
        )
        
        logger.info(
            f"Loaded run '{name}': {len(trajectories)} trajectories, "
            f"{len(resolved_ids)} resolved"
        )
        
        return run
    
    @staticmethod
    def load_results(results_path: Path | str) -> set[str]:
        """Load resolved instance IDs from SWE-bench results JSON.
        
        Args:
            results_path: Path to results JSON file
            
        Returns:
            Set of resolved instance IDs
        """
        path = Path(results_path)
        if not path.exists():
            logger.warning(f"Results file not found: {path}")
            return set()
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Handle different result formats
            if "resolved_ids" in data:
                return set(data["resolved_ids"])
            elif "resolved" in data:
                return set(data["resolved"])
            else:
                logger.warning(f"No resolved_ids found in {path}")
                return set()
        except Exception as e:
            logger.error(f"Error loading results from {path}: {e}")
            return set()
    
    @staticmethod
    def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
        """Load records from a JSONL file.
        
        Args:
            path: Path to JSONL file
            
        Returns:
            List of parsed JSON records
        """
        path = Path(path)
        records = []
        
        with open(path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in {path}: {e}")
        
        return records
    
    @staticmethod
    def detect_shell_command(cmd: str) -> str | None:
        """Extract the base shell command from a command string.
        
        Args:
            cmd: Full command string
            
        Returns:
            Base command (first word) or None
        """
        if not cmd:
            return None
        parts = cmd.strip().split()
        return parts[0] if parts else None

