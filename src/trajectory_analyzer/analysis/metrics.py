"""Metrics extraction from trajectories and runs."""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
import statistics

from ..models import Run, Trajectory


@dataclass
class RunMetrics:
    """Aggregated metrics for a single run."""
    
    run_name: str
    scaffold: str
    base_model: str
    lora_adapter: str | None
    
    # Instance counts
    total_instances: int = 0
    resolved_instances: int = 0
    instances_with_patch: int = 0
    
    # Rates
    resolve_rate: float = 0.0
    patch_rate: float = 0.0
    
    # Tool usage
    tool_counts: dict[str, int] = field(default_factory=dict)
    shell_command_counts: dict[str, int] = field(default_factory=dict)
    
    # Per-trajectory distributions
    tool_calls_per_traj: list[int] = field(default_factory=list)
    tokens_per_traj: list[int] = field(default_factory=list)
    steps_per_traj: list[int] = field(default_factory=list)
    
    # Success rate by tool
    tool_success_rates: dict[str, float] = field(default_factory=dict)
    
    # Resolved vs unresolved comparison
    resolved_avg_tool_calls: float = 0.0
    unresolved_avg_tool_calls: float = 0.0
    resolved_avg_tokens: float = 0.0
    unresolved_avg_tokens: float = 0.0
    
    @property
    def avg_tool_calls(self) -> float:
        """Average tool calls per trajectory."""
        if not self.tool_calls_per_traj:
            return 0.0
        return statistics.mean(self.tool_calls_per_traj)
    
    @property
    def median_tool_calls(self) -> float:
        """Median tool calls per trajectory."""
        if not self.tool_calls_per_traj:
            return 0.0
        return statistics.median(self.tool_calls_per_traj)
    
    @property
    def avg_tokens(self) -> float:
        """Average tokens per trajectory."""
        if not self.tokens_per_traj:
            return 0.0
        return statistics.mean(self.tokens_per_traj)
    
    @property
    def avg_steps(self) -> float:
        """Average steps per trajectory."""
        if not self.steps_per_traj:
            return 0.0
        return statistics.mean(self.steps_per_traj)
    
    @property
    def total_tool_calls(self) -> int:
        """Total tool calls across all trajectories."""
        return sum(self.tool_counts.values())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_name': self.run_name,
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'lora_adapter': self.lora_adapter,
            'total_instances': self.total_instances,
            'resolved_instances': self.resolved_instances,
            'instances_with_patch': self.instances_with_patch,
            'resolve_rate': self.resolve_rate,
            'patch_rate': self.patch_rate,
            'avg_tool_calls': self.avg_tool_calls,
            'median_tool_calls': self.median_tool_calls,
            'avg_tokens': self.avg_tokens,
            'avg_steps': self.avg_steps,
            'total_tool_calls': self.total_tool_calls,
            'tool_counts': self.tool_counts,
            'shell_command_counts': self.shell_command_counts,
            'tool_success_rates': self.tool_success_rates,
            'resolved_avg_tool_calls': self.resolved_avg_tool_calls,
            'unresolved_avg_tool_calls': self.unresolved_avg_tool_calls,
            'resolved_avg_tokens': self.resolved_avg_tokens,
            'unresolved_avg_tokens': self.unresolved_avg_tokens,
        }


class MetricsExtractor:
    """Extract metrics from runs and trajectories."""
    
    def extract_run_metrics(self, run: Run) -> RunMetrics:
        """Extract comprehensive metrics from a run.
        
        Args:
            run: Run to analyze
            
        Returns:
            RunMetrics with aggregated statistics
        """
        metrics = RunMetrics(
            run_name=run.name,
            scaffold=run.scaffold,
            base_model=run.base_model,
            lora_adapter=run.lora_adapter,
        )
        
        # Basic counts
        metrics.total_instances = len(run.trajectories)
        metrics.resolved_instances = run.num_resolved
        metrics.instances_with_patch = run.num_with_patch
        
        # Rates
        metrics.resolve_rate = run.resolve_rate
        metrics.patch_rate = run.patch_rate
        
        # Aggregate tool counts
        tool_counts: dict[str, int] = defaultdict(int)
        shell_counts: dict[str, int] = defaultdict(int)
        tool_successes: dict[str, int] = defaultdict(int)
        tool_totals: dict[str, int] = defaultdict(int)
        
        for traj in run.trajectories:
            # Per-trajectory metrics
            metrics.tool_calls_per_traj.append(traj.total_tool_calls)
            metrics.tokens_per_traj.append(traj.total_tokens)
            metrics.steps_per_traj.append(traj.num_steps)
            
            # Tool counts
            for tool, count in traj.get_tool_counts().items():
                tool_counts[tool] += count
            
            for cmd, count in traj.get_shell_command_counts().items():
                shell_counts[cmd] += count
            
            # Track success/failure for each tool
            for tc in traj.get_tool_calls():
                tool_totals[tc.name] += 1
                if tc.success:
                    tool_successes[tc.name] += 1
        
        metrics.tool_counts = dict(tool_counts)
        metrics.shell_command_counts = dict(shell_counts)
        
        # Compute tool success rates
        metrics.tool_success_rates = {
            tool: tool_successes[tool] / total
            for tool, total in tool_totals.items()
            if total > 0
        }
        
        # Resolved vs unresolved comparison
        resolved_trajs = run.get_resolved_trajectories()
        unresolved_trajs = run.get_unresolved_trajectories()
        
        if resolved_trajs:
            metrics.resolved_avg_tool_calls = statistics.mean(
                t.total_tool_calls for t in resolved_trajs
            )
            metrics.resolved_avg_tokens = statistics.mean(
                t.total_tokens for t in resolved_trajs
            )
        
        if unresolved_trajs:
            metrics.unresolved_avg_tool_calls = statistics.mean(
                t.total_tool_calls for t in unresolved_trajs
            )
            metrics.unresolved_avg_tokens = statistics.mean(
                t.total_tokens for t in unresolved_trajs
            )
        
        return metrics
    
    def extract_trajectory_metrics(self, traj: Trajectory) -> dict[str, Any]:
        """Extract metrics from a single trajectory.
        
        Args:
            traj: Trajectory to analyze
            
        Returns:
            Dictionary of metrics
        """
        return {
            'instance_id': traj.instance_id,
            'num_steps': traj.num_steps,
            'total_tool_calls': traj.total_tool_calls,
            'total_tokens': traj.total_tokens,
            'has_patch': traj.has_patch,
            'resolved': traj.resolved,
            'exit_reason': traj.exit_reason,
            'tool_counts': traj.get_tool_counts(),
            'shell_command_counts': traj.get_shell_command_counts(),
            'overall_success_rate': traj.get_tool_success_rate(),
        }
    
    def get_tool_distribution(self, run: Run) -> dict[str, int]:
        """Get tool call distribution for a run.
        
        Args:
            run: Run to analyze
            
        Returns:
            Dictionary mapping tool names to counts, sorted by count
        """
        counts = run.get_total_tool_counts()
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_shell_command_distribution(self, run: Run) -> dict[str, int]:
        """Get shell command distribution for a run.
        
        Args:
            run: Run to analyze
            
        Returns:
            Dictionary mapping shell commands to counts, sorted by count
        """
        counts = run.get_total_shell_command_counts()
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_success_correlation(
        self, 
        run: Run
    ) -> tuple[list[int], list[bool], list[int]]:
        """Get data for tool calls vs success correlation.
        
        Args:
            run: Run to analyze
            
        Returns:
            Tuple of (tool_calls, resolved, tokens) lists
        """
        tool_calls = []
        resolved = []
        tokens = []
        
        for traj in run.trajectories:
            if traj.resolved is not None:
                tool_calls.append(traj.total_tool_calls)
                resolved.append(traj.resolved)
                tokens.append(traj.total_tokens)
        
        return tool_calls, resolved, tokens

