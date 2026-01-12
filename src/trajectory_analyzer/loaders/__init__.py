"""Trajectory loaders for different agent formats."""

from .base import TrajectoryLoader
from .claude_code import ClaudeCodeLoader
from .nano_agent import NanoAgentLoader
from .r2e_gym import R2EGymLoader
from .swe_agent import SWEAgentLoader
from .swebench_instance import SWEBenchInstanceLoader

__all__ = [
    "TrajectoryLoader",
    "ClaudeCodeLoader",
    "NanoAgentLoader",
    "R2EGymLoader",
    "SWEAgentLoader",
    "SWEBenchInstanceLoader",
]

