"""Loader for R2E-Gym trajectory format."""

import logging
import re
from glob import glob
from pathlib import Path
from typing import Any

from .base import TrajectoryLoader
from ..models import Trajectory, Step, ToolCall

logger = logging.getLogger(__name__)


class R2EGymLoader(TrajectoryLoader):
    """Loader for R2E-Gym JSONL trajectory format.
    
    Expected format:
    - JSONL file with one record per instance
    - Each record has:
      - ds: dataset info with instance_id
      - trajectory_steps: list of steps
      - output_patch: generated patch
      - reward: 1.0 if resolved, 0.0 otherwise
      - exit_reason: why trajectory ended
    - Each step has:
      - action: XML-format function call
      - observation: tool result
      - thought: reasoning (may include <think> tags)
      - token_usage_total: tokens for this step
    """
    
    # Regex pattern for parsing XML-style function calls
    FUNCTION_PATTERN = re.compile(
        r'<function=(\w+)>(.*?)</function>',
        re.DOTALL
    )
    PARAMETER_PATTERN = re.compile(
        r'<parameter=(\w+)>(.*?)</parameter>',
        re.DOTALL
    )
    
    def load_trajectories(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from R2E-Gym JSONL file(s).
        
        Args:
            path: Path to JSONL file or glob pattern (e.g., 'trajs/*.jsonl')
            
        Returns:
            List of Trajectory objects
        """
        path_str = str(path)
        
        # Handle glob patterns
        if '*' in path_str:
            files = glob(path_str)
        else:
            files = [path_str]
        
        trajectories = []
        for file_path in files:
            file_trajs = self._load_file(Path(file_path))
            trajectories.extend(file_trajs)
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {len(files)} file(s)")
        return trajectories
    
    def _load_file(self, path: Path) -> list[Trajectory]:
        """Load trajectories from a single file."""
        records = self.load_jsonl(path)
        
        trajectories = []
        for record in records:
            try:
                traj = self._parse_record(record)
                trajectories.append(traj)
            except Exception as e:
                ds = record.get('ds', {})
                instance_id = ds.get('instance_id', 'unknown')
                logger.warning(f"Failed to parse trajectory for {instance_id}: {e}")
        
        return trajectories
    
    def _parse_record(self, record: dict[str, Any]) -> Trajectory:
        """Parse a single R2E-Gym record into a Trajectory."""
        ds = record.get('ds', {})
        instance_id = ds.get('instance_id', 'unknown')
        
        # Parse trajectory steps
        steps_data = record.get('trajectory_steps', [])
        steps = [self._parse_step(s, i) for i, s in enumerate(steps_data)]
        
        # Get resolution status from reward
        reward = record.get('reward', 0.0)
        resolved = reward >= 1.0 if reward is not None else None
        
        return Trajectory(
            instance_id=instance_id,
            steps=steps,
            generated_patch=record.get('output_patch'),
            resolved=resolved,
            exit_reason=record.get('exit_reason'),
        )
    
    def _parse_step(self, step_data: dict[str, Any], index: int) -> Step:
        """Parse a single step from R2E-Gym format."""
        action = step_data.get('action', '')
        observation = step_data.get('observation', '')
        thought = step_data.get('thought', '')
        
        # Clean up thought (remove <think> tags if present)
        if thought:
            thought = re.sub(r'</?think>', '', thought).strip()
        
        # Parse tool calls from action
        tool_calls = self._parse_action(action, observation)
        
        # Use completion tokens only (not total) to avoid counting cumulative prompts
        # token_usage_total includes prompt+completion, but prompts grow each step
        # and include previous context, so summing totals massively overcounts
        return Step(
            index=index,
            thought=thought if thought else None,
            tool_calls=tool_calls,
            token_usage=step_data.get('token_usage_completion', 0),
        )
    
    def _parse_action(self, action: str, observation: str) -> list[ToolCall]:
        """Parse XML-style function calls from action string."""
        tool_calls = []
        
        # Find all function calls in the action
        for match in self.FUNCTION_PATTERN.finditer(action):
            func_name = match.group(1)
            func_content = match.group(2)
            
            # Parse parameters
            arguments: dict[str, Any] = {}
            for param_match in self.PARAMETER_PATTERN.finditer(func_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                arguments[param_name] = param_value
            
            # Determine success based on observation
            success = self._determine_success(func_name, observation, arguments)
            
            # Extract shell command if applicable
            shell_command = None
            if func_name in ('execute_bash', 'bash', 'shell'):
                cmd = arguments.get('command') or arguments.get('cmd', '')
                shell_command = self.detect_shell_command(cmd)
            
            tool_calls.append(ToolCall(
                name=func_name,
                arguments=arguments,
                result=observation,
                success=success,
                shell_command=shell_command,
            ))
        
        return tool_calls
    
    def _determine_success(
        self,
        func_name: str,
        observation: str,
        arguments: dict[str, Any],
    ) -> bool:
        """Determine if a tool call was successful."""
        if not observation:
            return False
        
        obs_lower = observation.lower()
        
        # Check for common error patterns
        error_patterns = [
            'error:',
            'failed',
            'no such file',
            'not found',
            'command not found',
            'permission denied',
            'traceback',
            'exception:',
            'syntax error',
        ]
        
        for pattern in error_patterns:
            if pattern in obs_lower:
                return False
        
        # Check for exit code errors
        if 'exit code' in obs_lower:
            # Look for non-zero exit codes
            exit_match = re.search(r'exit code[:\s]+(\d+)', obs_lower)
            if exit_match and exit_match.group(1) != '0':
                return False
        
        return True
    
    def load_run(
        self,
        name: str,
        scaffold: str,
        base_model: str,
        trajectories_path: Path | str,
        results_path: Path | str | None = None,
        lora_adapter: str | None = None,
    ):
        """Load a run, using reward field as resolved status if no results file."""
        from ..models import Run
        
        trajectories = self.load_trajectories(trajectories_path)
        
        resolved_ids: set[str] = set()
        if results_path:
            resolved_ids = self.load_results(results_path)
        else:
            # Use reward field from trajectories
            resolved_ids = {
                t.instance_id for t in trajectories
                if t.resolved is True
            }
        
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

