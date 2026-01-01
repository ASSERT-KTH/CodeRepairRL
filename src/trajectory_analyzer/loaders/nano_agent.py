"""Loader for nano-agent trajectory format."""

import logging
from pathlib import Path
from typing import Any

from .base import TrajectoryLoader
from ..models import Trajectory, Step, ToolCall

logger = logging.getLogger(__name__)


class NanoAgentLoader(TrajectoryLoader):
    """Loader for nano-agent detailed_predictions.jsonl format.
    
    Expected format:
    - JSONL file with one record per instance
    - Each record has 'instance_id' and 'detailed_predictions' fields
    - detailed_predictions contains:
      - completion: list of chat messages (OpenAI format)
      - generated_diff: the patch output
      - token_usage: total tokens used
      - tool_usage: total tool calls
      - tool_calls_shell: number of shell tool calls
      - tool_calls_apply_patch: number of apply_patch tool calls
      - tool_success_rate_shell: success rate for shell
      - tool_success_rate_apply_patch: success rate for apply_patch
      - shell_cmd_*: counts for individual shell commands
    """
    
    def load_trajectories(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from a nano-agent JSONL file.
        
        Args:
            path: Path to detailed_predictions.jsonl
            
        Returns:
            List of Trajectory objects
        """
        path = Path(path)
        records = self.load_jsonl(path)
        
        trajectories = []
        for record in records:
            try:
                traj = self._parse_record(record)
                trajectories.append(traj)
            except Exception as e:
                instance_id = record.get('instance_id', 'unknown')
                logger.warning(f"Failed to parse trajectory for {instance_id}: {e}")
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return trajectories
    
    def _parse_record(self, record: dict[str, Any]) -> Trajectory:
        """Parse a single nano-agent record into a Trajectory."""
        instance_id = record.get('instance_id', 'unknown')
        det = record.get('detailed_predictions', {})
        
        # Parse chat completion into steps
        completion = det.get('completion', [])
        steps = self._parse_completion(completion, det)
        
        return Trajectory(
            instance_id=instance_id,
            steps=steps,
            generated_patch=det.get('generated_diff'),
            exit_reason=det.get('exit_reason'),
        )
    
    def _parse_completion(
        self, 
        completion: list[dict[str, Any]] | None, 
        det: dict[str, Any]
    ) -> list[Step]:
        """Parse OpenAI chat completion format into Steps.
        
        The completion is a list of alternating assistant/tool messages.
        Each assistant message with tool_calls forms a step.
        """
        steps: list[Step] = []
        step_index = 0
        
        if not completion:
            return steps
        
        i = 0
        while i < len(completion):
            msg = completion[i]
            role = msg.get('role')
            
            if role == 'assistant':
                # Parse assistant message as a step
                tool_calls_data = msg.get('tool_calls') or []
                content = msg.get('content') or ''
                
                # Collect tool results from subsequent tool messages
                tool_results: dict[str, str] = {}
                j = i + 1
                while j < len(completion) and completion[j].get('role') == 'tool':
                    tool_msg = completion[j]
                    tool_call_id = tool_msg.get('tool_call_id', '')
                    tool_result = tool_msg.get('content', '')
                    tool_results[tool_call_id] = tool_result
                    j += 1
                
                # Parse tool calls
                tool_calls = self._parse_tool_calls(tool_calls_data, tool_results, det)
                
                if tool_calls or content:  # Only add step if there's content
                    steps.append(Step(
                        index=step_index,
                        thought=content if content else None,
                        tool_calls=tool_calls,
                        token_usage=0,  # Token usage is aggregated in nano-agent format
                    ))
                    step_index += 1
                
                # Skip to after the tool messages
                i = j
            else:
                i += 1
        
        # Distribute total token usage across steps if available
        total_tokens = det.get('token_usage', 0)
        if steps and total_tokens > 0:
            tokens_per_step = total_tokens // len(steps)
            for step in steps:
                step.token_usage = tokens_per_step
            # Add remainder to last step
            steps[-1].token_usage += total_tokens % len(steps)
        
        return steps
    
    def _parse_tool_calls(
        self,
        tool_calls_data: list[dict[str, Any]],
        tool_results: dict[str, str],
        det: dict[str, Any],
    ) -> list[ToolCall]:
        """Parse tool calls from OpenAI format."""
        tool_calls = []
        
        # Get success rates for determining tool success
        shell_success_rate = det.get('tool_success_rate_shell', 1.0)
        apply_patch_success_rate = det.get('tool_success_rate_apply_patch', 1.0)
        
        for tc_data in tool_calls_data:
            func_data = tc_data.get('function', {})
            tool_name = func_data.get('name', 'unknown')
            
            # Parse arguments
            args_str = func_data.get('arguments', '{}')
            try:
                import json
                arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                arguments = {'raw': args_str}
            
            # Get result from tool_results mapping
            tool_call_id = tc_data.get('id', '')
            result = tool_results.get(tool_call_id)
            
            # Determine success based on result content
            success = self._determine_success(tool_name, result, arguments)
            
            # Extract shell command if applicable
            shell_command = None
            if tool_name in ('shell', 'execute_bash', 'bash'):
                cmd = arguments.get('cmd') or arguments.get('command', '')
                shell_command = self.detect_shell_command(cmd)
            
            tool_calls.append(ToolCall(
                name=tool_name,
                arguments=arguments,
                result=result,
                success=success,
                shell_command=shell_command,
            ))
        
        return tool_calls
    
    def _determine_success(
        self, 
        tool_name: str, 
        result: str | None,
        arguments: dict[str, Any],
    ) -> bool:
        """Determine if a tool call was successful based on its result."""
        if result is None:
            return False
        
        result_lower = result.lower()
        
        # Check for common error patterns
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
            if pattern in result_lower:
                return False
        
        # For apply_patch, check for specific failure indicators
        if tool_name == 'apply_patch':
            if 'patch failed' in result_lower or 'could not apply' in result_lower:
                return False
            if 'not unique' in result_lower or 'multiple matches' in result_lower:
                return False
        
        return True

