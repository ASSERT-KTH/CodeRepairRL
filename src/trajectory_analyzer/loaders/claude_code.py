"""Loader for Claude Code trajectory format."""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Any

from .base import TrajectoryLoader
from ..models import Trajectory, Step, ToolCall

logger = logging.getLogger(__name__)


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int, handling None."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


class ClaudeCodeLoader(TrajectoryLoader):
    """Loader for Claude Code JSONL trajectory format.
    
    Expected format:
    - JSONL file(s) with one JSON object per line
    - Each line represents a message or event
    - Messages have:
      - type: "user", "assistant", "queue-operation", etc.
      - message: contains role and content
      - sessionId: groups messages into trajectories
      - uuid: unique message identifier
    - Assistant messages can contain tool_use items in content
    - Tool results come as user messages with tool_use_id
    """
    
    def load_trajectories(self, path: Path | str) -> list[Trajectory]:
        """Load trajectories from Claude Code JSONL file(s).
        
        Args:
            path: Path to JSONL file or glob pattern (e.g., '**/*.jsonl')
            
        Returns:
            List of Trajectory objects
        """
        path_str = str(path)
        
        # Handle glob patterns
        if '*' in path_str:
            files = glob(path_str, recursive=True)
        else:
            files = [path_str]
        
        if not files:
            logger.warning(f"No files found matching pattern: {path_str}")
            return []
        
        # Group messages by sessionId
        sessions: dict[str, list[dict[str, Any]]] = {}
        
        for file_path in files:
            try:
                records = self.load_jsonl(Path(file_path))
                file_path_obj = Path(file_path)
                
                # Extract instance_id from file path (e.g., .../instance_id/projects/default/file.jsonl)
                instance_id_from_path = None
                parts = file_path_obj.parts
                for i, part in enumerate(parts):
                    if part == 'projects' and i > 0:
                        instance_id_from_path = parts[i - 1]
                        break
                
                for record in records:
                    session_id = record.get('sessionId')
                    if not session_id:
                        session_id = file_path_obj.stem
                    
                    # Store instance_id in record for later use
                    if instance_id_from_path:
                        record['_instance_id'] = instance_id_from_path
                    
                    if session_id not in sessions:
                        sessions[session_id] = []
                    sessions[session_id].append(record)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        # Parse each session into a trajectory
        trajectories = []
        for session_id, messages in sessions.items():
            try:
                traj = self._parse_session(session_id, messages)
                if traj:
                    trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Failed to parse session {session_id}: {e}")
        
        logger.info(f"Loaded {len(trajectories)} trajectories from {len(files)} file(s)")
        return trajectories
    
    def _parse_session(self, session_id: str, messages: list[dict[str, Any]]) -> Trajectory | None:
        """Parse a session (group of messages) into a Trajectory.
        
        Args:
            session_id: Session identifier (used as instance_id)
            messages: List of message records
            
        Returns:
            Trajectory object or None if parsing fails
        """
        # Sort messages by timestamp if available
        try:
            messages = sorted(
                messages,
                key=lambda m: m.get('timestamp', ''),
                reverse=False
            )
        except Exception:
            pass  # Keep original order if sorting fails
        
        steps: list[Step] = []
        step_index = 0
        
        # Track tool results by tool_use_id
        tool_results: dict[str, str] = {}
        
        # Track current step's tool calls
        current_tool_calls: list[ToolCall] = []
        current_thought: str | None = None
        current_token_usage = 0
        current_input_tokens = 0
        current_output_tokens = 0
        current_cache_read_tokens = 0
        
        i = 0
        while i < len(messages):
            msg = messages[i]
            msg_type = msg.get('type', '')
            message = msg.get('message', {})
            
            if msg_type == 'assistant':
                # Assistant message - may contain tool calls
                content = message.get('content', [])
                usage = msg.get('message', {}).get('usage', {})
                
                # Extract token usage (handle None values)
                current_input_tokens = safe_int(usage.get('input_tokens'), 0)
                current_output_tokens = safe_int(usage.get('output_tokens'), 0)
                current_cache_read_tokens = safe_int(usage.get('cache_read_input_tokens'), 0)
                current_cache_creation_tokens = safe_int(usage.get('cache_creation_input_tokens'), 0)
                current_token_usage = current_input_tokens + current_output_tokens
                
                # Extract text content (thought)
                text_parts = []
                tool_uses = []
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'tool_use':
                                tool_uses.append(item)
                elif isinstance(content, str):
                    text_parts.append(content)
                
                current_thought = ' '.join(text_parts).strip() or None
                
                # Parse tool calls and collect their IDs
                current_tool_calls = []
                tool_use_ids = []
                for tool_use in tool_uses:
                    tool_use_id = tool_use.get('id', '')
                    tool_use_ids.append(tool_use_id)
                    tool_call = self._parse_tool_use(tool_use, {})  # Parse without results first
                    if tool_call:
                        current_tool_calls.append(tool_call)
                
                # Look ahead for tool results
                j = i + 1
                while j < len(messages):
                    next_msg = messages[j]
                    if next_msg.get('type') == 'user':
                        next_message = next_msg.get('message', {})
                        next_content = next_message.get('content', [])
                        
                        # Check for tool results
                        if isinstance(next_content, list):
                            for item in next_content:
                                if isinstance(item, dict) and item.get('type') == 'tool_result':
                                    tool_use_id = item.get('tool_use_id', '')
                                    tool_result = item.get('content', '')
                                    if isinstance(tool_result, str):
                                        tool_results[tool_use_id] = tool_result
                                    elif isinstance(tool_result, dict):
                                        tool_results[tool_use_id] = json.dumps(tool_result)
                        elif isinstance(next_content, dict):
                            tool_use_id = next_content.get('tool_use_id', '')
                            tool_result = next_content.get('content', '')
                            if tool_use_id:
                                if isinstance(tool_result, str):
                                    tool_results[tool_use_id] = tool_result
                                elif isinstance(tool_result, dict):
                                    tool_results[tool_use_id] = json.dumps(tool_result)
                        
                        # Stop if we hit another assistant message
                        if next_msg.get('type') == 'assistant':
                            break
                    elif next_msg.get('type') == 'assistant':
                        break
                    j += 1
                
                # Update tool calls with results using tool_use_id
                for idx, tool_call in enumerate(current_tool_calls):
                    if idx < len(tool_use_ids):
                        tool_use_id = tool_use_ids[idx]
                        tool_call.result = tool_results.get(tool_use_id)
                
                # Create step if there's content or tool calls
                if current_thought or current_tool_calls:
                    
                    steps.append(Step(
                        index=step_index,
                        thought=current_thought,
                        tool_calls=current_tool_calls,
                        token_usage=current_token_usage,
                        input_tokens=current_input_tokens,
                        output_tokens=current_output_tokens,
                        cache_read_input_tokens=current_cache_read_tokens,
                        cache_creation_input_tokens=current_cache_creation_tokens,
                    ))
                    step_index += 1
                
                i = j  # Skip to after tool results
            else:
                i += 1
        
        # Extract instance_id from messages or session_id
        instance_id = session_id
        
        # Try to extract from messages (stored during file loading)
        if messages:
            first_msg = messages[0]
            if '_instance_id' in first_msg:
                instance_id = first_msg['_instance_id']
            else:
                # Try to infer from cwd
                cwd = first_msg.get('cwd', '')
                if cwd:
                    # Extract repo name from path like /repo-name
                    parts = cwd.strip('/').split('/')
                    if parts:
                        repo_name = parts[-1]
                        # If session_id is a UUID, use repo_name as instance_id
                        if len(session_id) > 20:  # Likely a UUID
                            instance_id = repo_name
        
        return Trajectory(
            instance_id=instance_id,
            steps=steps,
            generated_patch=None,  # Claude Code format doesn't include patches in trajectory
            exit_reason=None,
        )
    
    def _parse_tool_use(self, tool_use: dict[str, Any], tool_results: dict[str, str] | None = None) -> ToolCall | None:
        """Parse a tool_use item into a ToolCall.
        
        Args:
            tool_use: Tool use dictionary from message content
            tool_results: Optional dictionary mapping tool_use_id to results
            
        Returns:
            ToolCall object or None
        """
        tool_name = tool_use.get('name', 'unknown')
        tool_id = tool_use.get('id', '')
        tool_input = tool_use.get('input', {})
        
        # Get result if available
        result = tool_results.get(tool_id) if tool_results else None
        
        # Parse arguments
        if isinstance(tool_input, dict):
            arguments = tool_input
        elif isinstance(tool_input, str):
            try:
                arguments = json.loads(tool_input)
            except json.JSONDecodeError:
                arguments = {'raw': tool_input}
        else:
            arguments = {}
        
        # Determine success
        success = self._determine_success(tool_name, result, arguments)
        
        # Extract shell command if applicable
        shell_command = None
        if tool_name.lower() in ('shell', 'bash', 'execute', 'exec', 'run_command', 'command'):
            cmd = arguments.get('cmd') or arguments.get('command') or arguments.get('input', '')
            if isinstance(cmd, str):
                shell_command = self.detect_shell_command(cmd)
        
        return ToolCall(
            name=tool_name,
            arguments=arguments,
            result=result,
            success=success,
            shell_command=shell_command,
        )
    
    def _determine_success(
        self, 
        tool_name: str, 
        result: str | None,
        arguments: dict[str, Any],
    ) -> bool:
        """Determine if a tool call was successful based on its result."""
        if result is None:
            return False
        
        result_lower = str(result).lower()
        
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
            'valueerror',
            'typeerror',
            'attributeerror',
        ]
        
        for pattern in error_patterns:
            if pattern in result_lower:
                return False
        
        return True
    
    def load_results(self, results_path: Path | str) -> set[str]:
        """Load resolved instance IDs from Claude Code results JSON.
        
        Claude Code results format is a list of objects with:
        - instance_id: the instance identifier
        - accuracy: boolean indicating if resolved
        
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
            
            # Handle list format (Claude Code format)
            if isinstance(data, list):
                resolved = set()
                for item in data:
                    if isinstance(item, dict):
                        instance_id = item.get('instance_id')
                        accuracy = item.get('accuracy', False)
                        if instance_id and accuracy:
                            resolved.add(instance_id)
                return resolved
            # Fall back to base class format
            elif "resolved_ids" in data:
                return set(data["resolved_ids"])
            elif "resolved" in data:
                return set(data["resolved"])
            else:
                logger.warning(f"No resolved_ids found in {path}")
                return set()
        except Exception as e:
            logger.error(f"Error loading results from {path}: {e}")
            return set()

