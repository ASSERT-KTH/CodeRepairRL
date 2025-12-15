"""Loader for SWE-agent log file format."""

import logging
import re
from pathlib import Path
from typing import Any

from .base import TrajectoryLoader
from ..models import Trajectory, Step, ToolCall

logger = logging.getLogger(__name__)


class SWEAgentLoader(TrajectoryLoader):
    """Loader for SWE-agent log file format.
    
    Expected format:
    - Directory containing subdirectories for each instance
    - Each subdirectory contains .log files
    - Log files contain "🎬 ACTION" markers to identify tool calls
    - Tool name is the first word after the ACTION marker
    """
    
    # Patterns for parsing SWE-agent logs
    ACTION_MARKER = "🎬 ACTION"
    OBSERVATION_MARKER = "🔎 OBSERVATION"
    THOUGHT_MARKER = "💭 THOUGHT"
    
    def load_trajectories(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from a SWE-agent logs directory.
        
        Args:
            path: Path to directory containing instance subdirectories
            
        Returns:
            List of Trajectory objects
        """
        logs_dir = Path(path)
        
        if not logs_dir.exists():
            raise FileNotFoundError(f"Log directory not found: {logs_dir}")
        
        trajectories = []
        
        # Find all subdirectories (each represents an instance)
        for instance_dir in sorted(logs_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            
            instance_id = instance_dir.name
            
            # Find .log files in the subdirectory
            log_files = list(instance_dir.glob("*.log"))
            if not log_files:
                continue
            
            # Use the first .log file found
            log_file = log_files[0]
            
            try:
                traj = self._load_log_file(log_file, instance_id)
                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Failed to load {log_file}: {e}")
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {logs_dir}")
        return trajectories
    
    def _load_log_file(self, log_file: Path, instance_id: str) -> Trajectory:
        """Load a single log file into a Trajectory."""
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        steps = self._parse_log_content(log_content)
        
        # Try to extract patch from log (if present)
        generated_patch = self._extract_patch(log_content)
        
        return Trajectory(
            instance_id=instance_id,
            steps=steps,
            generated_patch=generated_patch,
        )
    
    def _parse_log_content(self, log_content: str) -> list[Step]:
        """Parse log content into Steps."""
        lines = log_content.split('\n')
        steps: list[Step] = []
        
        current_thought: str | None = None
        current_action: str | None = None
        current_observation: str | None = None
        step_index = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            if self.THOUGHT_MARKER in line:
                # Start collecting thought
                current_thought = self._extract_section(lines, i + 1)
                
            elif self.ACTION_MARKER in line:
                # Extract action
                if i + 1 < len(lines):
                    current_action = lines[i + 1].strip()
                else:
                    # Check if action is on same line
                    parts = line.split(self.ACTION_MARKER, 1)
                    if len(parts) > 1:
                        current_action = parts[1].strip()
                
            elif self.OBSERVATION_MARKER in line:
                # Extract observation and complete the step
                current_observation = self._extract_section(lines, i + 1)
                
                # Create step if we have an action
                if current_action:
                    tool_calls = self._parse_action(current_action, current_observation)
                    
                    steps.append(Step(
                        index=step_index,
                        thought=current_thought,
                        tool_calls=tool_calls,
                        token_usage=0,  # Not available in log format
                    ))
                    step_index += 1
                
                # Reset for next step
                current_thought = None
                current_action = None
                current_observation = None
            
            i += 1
        
        # Handle any remaining action without observation
        if current_action:
            tool_calls = self._parse_action(current_action, None)
            steps.append(Step(
                index=step_index,
                thought=current_thought,
                tool_calls=tool_calls,
                token_usage=0,
            ))
        
        return steps
    
    def _extract_section(self, lines: list[str], start_idx: int) -> str:
        """Extract a section of text until the next marker."""
        section_lines = []
        markers = [self.THOUGHT_MARKER, self.ACTION_MARKER, self.OBSERVATION_MARKER]
        
        i = start_idx
        while i < len(lines):
            line = lines[i]
            # Stop if we hit another marker
            if any(marker in line for marker in markers):
                break
            section_lines.append(line)
            i += 1
        
        return '\n'.join(section_lines).strip()
    
    def _parse_action(
        self, 
        action: str, 
        observation: str | None
    ) -> list[ToolCall]:
        """Parse action string into tool calls."""
        tool_calls = []
        
        if not action:
            return tool_calls
        
        # Get tool name (first word)
        words = action.split()
        if not words:
            return tool_calls
        
        tool_name = words[0].strip()
        
        # Parse arguments (everything after tool name)
        args_str = ' '.join(words[1:]) if len(words) > 1 else ''
        arguments = self._parse_arguments(tool_name, args_str)
        
        # Determine success based on observation
        success = self._determine_success(tool_name, observation)
        
        # Extract shell command if applicable
        shell_command = None
        if tool_name.lower() in ('bash', 'run_command', 'execute', 'run', 'shell'):
            cmd = arguments.get('command') or arguments.get('cmd', args_str)
            shell_command = self.detect_shell_command(cmd)
        
        tool_calls.append(ToolCall(
            name=tool_name,
            arguments=arguments,
            result=observation,
            success=success,
            shell_command=shell_command,
        ))
        
        return tool_calls
    
    def _parse_arguments(self, tool_name: str, args_str: str) -> dict[str, Any]:
        """Parse argument string into a dictionary."""
        if not args_str:
            return {}
        
        # Try to detect key=value patterns
        args: dict[str, Any] = {}
        
        # Check for common patterns
        # Pattern 1: key=value pairs
        kv_pattern = re.findall(r'(\w+)=(["\']?)(.+?)\2(?:\s|$)', args_str)
        if kv_pattern:
            for key, _, value in kv_pattern:
                args[key] = value
        
        # Pattern 2: Just a command/path
        if not args:
            if tool_name.lower() in ('bash', 'run_command', 'execute', 'run', 'shell'):
                args['command'] = args_str
            else:
                args['raw'] = args_str
        
        return args
    
    def _determine_success(self, tool_name: str, observation: str | None) -> bool:
        """Determine if a tool call was successful."""
        if observation is None:
            return True  # Assume success if no observation
        
        obs_lower = observation.lower()
        
        error_patterns = [
            'error:',
            'failed',
            'no such file',
            'not found',
            'command not found',
            'permission denied',
            'traceback',
            'exception',
        ]
        
        for pattern in error_patterns:
            if pattern in obs_lower:
                return False
        
        return True
    
    def _extract_patch(self, log_content: str) -> str | None:
        """Try to extract generated patch from log content."""
        # Look for common patch markers
        patch_markers = [
            'diff --git',
            '--- a/',
            '+++ b/',
        ]
        
        for marker in patch_markers:
            if marker in log_content:
                # Try to extract the diff section
                start = log_content.find('diff --git')
                if start == -1:
                    start = log_content.find('--- a/')
                
                if start != -1:
                    # Find the end of the patch (look for common end patterns)
                    end = len(log_content)
                    for end_marker in ['🎬', '💭', '🔎', '\n\n\n']:
                        pos = log_content.find(end_marker, start)
                        if pos != -1 and pos < end:
                            end = pos
                    
                    patch = log_content[start:end].strip()
                    if patch:
                        return patch
        
        return None

