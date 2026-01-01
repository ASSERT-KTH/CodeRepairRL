"""Run model representing a collection of trajectories with metadata."""

from dataclasses import dataclass, field
from typing import Iterator
from collections import defaultdict

from .trajectory import Trajectory


@dataclass
class Run:
    """Represents an evaluation run: a collection of trajectories with metadata."""
    
    name: str
    """Run identifier, e.g., 'nano-agent-DeepSWE-Preview'."""
    
    scaffold: str
    """Agent scaffold type: 'nano-agent', 'r2e-gym', or 'swe-agent'."""
    
    base_model: str
    """Base model identifier, e.g., 'Qwen/Qwen3-32B'."""
    
    trajectories: list[Trajectory] = field(default_factory=list)
    """List of trajectories in this run."""
    
    resolved_ids: set[str] = field(default_factory=set)
    """Instance IDs that were resolved (from SWE-bench results)."""
    
    lora_adapter: str | None = None
    """LoRA adapter name if applicable, e.g., 'GSPO-Final-lora-step-460'."""
    
    def __post_init__(self):
        # Apply resolved status to trajectories if we have results
        if self.resolved_ids:
            for traj in self.trajectories:
                traj.resolved = traj.instance_id in self.resolved_ids
    
    def add_trajectory(self, trajectory: Trajectory) -> None:
        """Add a trajectory and update its resolved status."""
        if self.resolved_ids:
            trajectory.resolved = trajectory.instance_id in self.resolved_ids
        self.trajectories.append(trajectory)
    
    def set_resolved_ids(self, resolved_ids: set[str]) -> None:
        """Set resolved IDs and update all trajectories."""
        self.resolved_ids = resolved_ids
        for traj in self.trajectories:
            traj.resolved = traj.instance_id in self.resolved_ids
    
    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self.trajectories)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    @property
    def num_instances(self) -> int:
        """Number of instances in this run."""
        return len(self.trajectories)
    
    @property
    def num_resolved(self) -> int:
        """Number of resolved instances."""
        return sum(1 for t in self.trajectories if t.resolved)
    
    @property
    def num_with_patch(self) -> int:
        """Number of instances where agent generated a patch."""
        return sum(1 for t in self.trajectories if t.has_patch)
    
    @property
    def resolve_rate(self) -> float:
        """Resolution rate (resolved / total)."""
        if not self.trajectories:
            return 0.0
        return self.num_resolved / len(self.trajectories)
    
    @property
    def patch_rate(self) -> float:
        """Patch generation rate (has_patch / total)."""
        if not self.trajectories:
            return 0.0
        return self.num_with_patch / len(self.trajectories)
    
    @property
    def is_finetuned(self) -> bool:
        """Whether this run uses a fine-tuned model."""
        return self.lora_adapter is not None
    
    def get_total_tool_counts(self) -> dict[str, int]:
        """Aggregate tool counts across all trajectories."""
        counts: dict[str, int] = defaultdict(int)
        for traj in self.trajectories:
            for tool, count in traj.get_tool_counts().items():
                counts[tool] += count
        return dict(counts)
    
    def get_total_shell_command_counts(self) -> dict[str, int]:
        """Aggregate shell command counts across all trajectories."""
        counts: dict[str, int] = defaultdict(int)
        for traj in self.trajectories:
            for cmd, count in traj.get_shell_command_counts().items():
                counts[cmd] += count
        return dict(counts)
    
    def get_avg_tool_calls(self) -> float:
        """Average tool calls per trajectory."""
        if not self.trajectories:
            return 0.0
        return sum(t.total_tool_calls for t in self.trajectories) / len(self.trajectories)
    
    def get_avg_tokens(self) -> float:
        """Average tokens per trajectory."""
        if not self.trajectories:
            return 0.0
        return sum(t.total_tokens for t in self.trajectories) / len(self.trajectories)
    
    def get_avg_steps(self) -> float:
        """Average steps per trajectory."""
        if not self.trajectories:
            return 0.0
        return sum(t.num_steps for t in self.trajectories) / len(self.trajectories)
    
    def get_resolved_trajectories(self) -> list[Trajectory]:
        """Get trajectories that were resolved."""
        return [t for t in self.trajectories if t.resolved]
    
    def get_unresolved_trajectories(self) -> list[Trajectory]:
        """Get trajectories that were not resolved."""
        return [t for t in self.trajectories if t.resolved is False]
    
    def get_unknown_trajectories(self) -> list[Trajectory]:
        """Get trajectories with unknown resolution status."""
        return [t for t in self.trajectories if t.resolved is None]

