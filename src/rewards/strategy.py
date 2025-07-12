import re
import json
from typing import Iterator, Any

# Tool call parsing pattern
tool_call_pattern = re.compile(
    r'<tool_call>\s*\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*({[^}]+})\s*\}</tool_call>'
    r'.*?<tool_response>(.*?)</tool_response>',
    re.DOTALL
)

def parse_tool_calls(completion: str) -> Iterator[tuple[str, bool, dict[str, Any]]]:
    """Extract tool calls from completion, focusing on first 5 calls."""
    matches = list(tool_call_pattern.finditer(completion))[:5]
    
    for match in matches:
        tool_name = match.group(1)
        tool_args = match.group(2)
        tool_response = match.group(3)

        if tool_name == "shell":
            try:
                tool_args = json.loads(tool_args)
                cmd = tool_args.get("cmd", "")
                success = not ("command failed" in tool_response or "command timed out" in tool_response)
                yield cmd, success, tool_args
            except json.JSONDecodeError:
                pass

def get_shell_commands(completion: str, n: int = 5) -> list[str]:
    """Get first n successful shell commands from completion."""
    commands = []
    for cmd, success, _ in parse_tool_calls(completion):
        if success and cmd:
            commands.append(cmd)
        if len(commands) >= n:
            break
    return commands

# Command categories by depth/specificity
OVERVIEW_COMMANDS = ["ls", "ls -la", "ls -l", "pwd"]
DISCOVERY_COMMANDS = ["find", "rg -l", "grep -l", "find . -name", "find . -type"]
SEARCH_COMMANDS = ["rg", "grep", "rg -n", "grep -n"]
EXAMINATION_COMMANDS = ["head", "tail", "cat"]
DETAIL_COMMANDS = ["rg -A", "rg -B", "rg -C", "grep -A", "grep -B", "grep -C"]
ANALYSIS_COMMANDS = ["wc", "sort", "uniq", "awk", "sed", "cut"]

# Define exploration strategies as sequences of command types
# Each strategy is a list of command sets representing a valid progression
STRATEGIES = {
    # Simple 2-step strategies (low-hanging fruit)
    "overview_examine": {
        "sequence": [OVERVIEW_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 0.4
    },
    "overview_discover": {
        "sequence": [OVERVIEW_COMMANDS, DISCOVERY_COMMANDS],
        "reward": 0.5
    },
    "discover_examine": {
        "sequence": [DISCOVERY_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 0.6
    },
    "search_examine": {
        "sequence": [SEARCH_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 0.65
    },
    
    # 3-step strategies
    "overview_discover_examine": {
        "sequence": [OVERVIEW_COMMANDS, DISCOVERY_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 0.8
    },
    "discover_search_detail": {
        "sequence": [DISCOVERY_COMMANDS, SEARCH_COMMANDS, DETAIL_COMMANDS],
        "reward": 0.9
    },
    "structure_search_examine": {
        "sequence": [OVERVIEW_COMMANDS, SEARCH_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 0.7
    },
    "search_examine_detail": {
        "sequence": [SEARCH_COMMANDS, EXAMINATION_COMMANDS, DETAIL_COMMANDS],
        "reward": 0.85
    },
    
    # 4-step strategy (comprehensive)
    "overview_discover_search_examine": {
        "sequence": [OVERVIEW_COMMANDS, DISCOVERY_COMMANDS, SEARCH_COMMANDS, EXAMINATION_COMMANDS],
        "reward": 1.0
    },
}

def command_matches_category(cmd: str, category: list[str]) -> bool:
    """Check if a command matches any pattern in a category."""
    cmd_base = cmd.strip().split()[0] if cmd else ""
    
    # Direct match
    if cmd_base in category:
        return True
    
    # Pattern match (for commands with specific flags/patterns)
    for pattern in category:
        if pattern in cmd:
            return True
    
    return False

def evaluate_strategy(commands: list[str], sequence: list[list[str]]) -> float:
    """
    Evaluate if commands follow a specific strategy sequence.
    Returns 1.0 if sequence is found, 0.0 otherwise.
    """
    if not commands or not sequence:
        return 0.0
    
    seq_index = 0
    for cmd in commands:
        if seq_index >= len(sequence):
            break
            
        # Check if command matches current sequence step
        if command_matches_category(cmd, sequence[seq_index]):
            seq_index += 1
    
    # Return 1.0 if we completed the sequence
    return 1.0 if seq_index >= len(sequence) else 0.0

def progressive_understanding_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward progressive understanding patterns in first 5 commands.
    Evaluates multiple strategies and returns normalized rewards 0-1.
    """
    rewards = []
    
    for completion in completions:
        # Extract content from completion structure
        content = completion[0]["content"] if completion else ""
        commands = get_shell_commands(content, 5)
        if not commands:
            rewards.append(0.0)
            continue
        
        # Evaluate all strategies
        max_reward = 0.0
        for strategy_info in STRATEGIES.values():
            sequence = strategy_info["sequence"]
            base_reward = strategy_info["reward"]
            
            if evaluate_strategy(commands, sequence):
                max_reward = max(max_reward, base_reward)
        
        rewards.append(max_reward)
    
    return rewards

# Token efficient, useful, and safe commands
DESIRABLE_COMMANDS = ["ls", "rg", "grep", "find", "head", "tail", "cat", "pwd", "wc", "sort", "uniq"]

def command_preference_reward_func(completions, **kwargs) -> list[float]:
    """Reward desirable commands, penalize anything else (0-1 scale)."""
    rewards = []
    
    for completion in completions:
        # Extract content from completion structure
        content = completion[0]["content"] if completion else ""
        commands = get_shell_commands(content, 5)
        if not commands:
            rewards.append(0.0)
            continue
        
        score = 0.0
        for cmd in commands:
            cmd_base = cmd.strip().split()[0] if cmd else ""
            
            if cmd_base in DESIRABLE_COMMANDS:
                score += 0.2  # Max 1.0 for 5 good commands
            else:
                score -= 0.1  # Penalty for any non-desirable command
        
        rewards.append(max(0.0, min(1.0, score)))  # Clamp to [0, 1]
    
    return rewards
