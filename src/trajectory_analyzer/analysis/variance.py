"""Variance analysis across runs for the same scaffold-model pairs."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from ..models import Run, Instance
from .metrics import MetricsExtractor, RunMetrics

logger = logging.getLogger(__name__)


@dataclass
class InstanceVarianceMetrics:
    """Variance metrics for a single instance across multiple runs."""
    
    instance_id: str
    scaffold: str
    base_model: str
    
    # Resolution outcomes across runs
    resolved_outcomes: list[bool] = field(default_factory=list)
    """List of resolved outcomes (True/False) for each run."""
    
    # Computed metrics
    resolve_rate: float = 0.0
    """Mean resolution rate across runs."""
    
    resolve_variance: float = 0.0
    """Variance in resolution outcomes (0.0 to 0.25 max for binary outcomes)."""
    
    resolve_stddev: float = 0.0
    """Standard deviation of resolution outcomes."""
    
    num_runs: int = 0
    """Number of runs this instance appears in."""
    
    # Patch generation outcomes
    patch_outcomes: list[bool] = field(default_factory=list)
    """List of patch generation outcomes for each run."""
    
    patch_rate: float = 0.0
    """Mean patch generation rate across runs."""
    
    # Instance difficulty metrics (from Instance object)
    num_files_changed: int | None = None
    num_lines_changed: int | None = None
    num_additions: int | None = None
    num_deletions: int | None = None
    
    def __post_init__(self):
        """Compute variance metrics from outcomes."""
        if self.resolved_outcomes:
            self.num_runs = len(self.resolved_outcomes)
            self.resolve_rate = np.mean(self.resolved_outcomes)
            self.resolve_variance = np.var(self.resolved_outcomes, ddof=0)
            self.resolve_stddev = np.std(self.resolved_outcomes, ddof=0)
        
        if self.patch_outcomes:
            self.patch_rate = np.mean(self.patch_outcomes)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'resolve_rate': self.resolve_rate,
            'resolve_variance': self.resolve_variance,
            'resolve_stddev': self.resolve_stddev,
            'num_runs': self.num_runs,
            'patch_rate': self.patch_rate,
            'num_files_changed': self.num_files_changed,
            'num_lines_changed': self.num_lines_changed,
            'num_additions': self.num_additions,
            'num_deletions': self.num_deletions,
        }


@dataclass
class RunMetricsStatistics:
    """Statistics for a metric across multiple runs."""
    
    mean: float = 0.0
    stddev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    values: list[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute statistics from values."""
        if self.values:
            self.mean = np.mean(self.values)
            self.stddev = np.std(self.values, ddof=1) if len(self.values) > 1 else 0.0
            self.min = np.min(self.values)
            self.max = np.max(self.values)


@dataclass
class DeepVarianceAnalysis:
    """Deep variance analysis with mean/stddev for all metrics across runs."""
    
    scaffold: str
    base_model: str
    num_runs: int = 0
    
    # Resolution metrics
    resolve_rate: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    patch_rate: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Instance counts
    total_instances: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    resolved_instances: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    instances_with_patch: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Tool usage metrics
    avg_tool_calls: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    median_tool_calls: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    total_tool_calls: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Token metrics
    avg_tokens: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    avg_input_tokens: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    avg_output_tokens: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Step metrics
    avg_steps: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Resolved vs unresolved comparison
    resolved_avg_tool_calls: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    unresolved_avg_tool_calls: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    resolved_avg_tokens: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    unresolved_avg_tokens: RunMetricsStatistics = field(default_factory=RunMetricsStatistics)
    
    # Tool success rates (aggregated across all runs)
    tool_success_rates: dict[str, RunMetricsStatistics] = field(default_factory=dict)
    
    # Tool usage counts (aggregated across all runs)
    tool_counts_mean: dict[str, float] = field(default_factory=dict)
    tool_counts_stddev: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'num_runs': self.num_runs,
            'resolve_rate': {'mean': self.resolve_rate.mean, 'stddev': self.resolve_rate.stddev, 'min': self.resolve_rate.min, 'max': self.resolve_rate.max},
            'patch_rate': {'mean': self.patch_rate.mean, 'stddev': self.patch_rate.stddev, 'min': self.patch_rate.min, 'max': self.patch_rate.max},
            'avg_tool_calls': {'mean': self.avg_tool_calls.mean, 'stddev': self.avg_tool_calls.stddev, 'min': self.avg_tool_calls.min, 'max': self.avg_tool_calls.max},
            'avg_tokens': {'mean': self.avg_tokens.mean, 'stddev': self.avg_tokens.stddev, 'min': self.avg_tokens.min, 'max': self.avg_tokens.max},
            'avg_steps': {'mean': self.avg_steps.mean, 'stddev': self.avg_steps.stddev, 'min': self.avg_steps.min, 'max': self.avg_steps.max},
        }


@dataclass
class VarianceAnalysis:
    """Variance analysis results for a scaffold-model pair."""
    
    scaffold: str
    base_model: str
    
    # Per-instance metrics
    instance_metrics: list[InstanceVarianceMetrics] = field(default_factory=list)
    
    # Correlation results
    correlation_files_resolve_rate: float | None = None
    """Correlation between num_files_changed and resolve_rate."""
    
    correlation_lines_resolve_rate: float | None = None
    """Correlation between num_lines_changed and resolve_rate."""
    
    correlation_files_variance: float | None = None
    """Correlation between num_files_changed and resolve_variance."""
    
    correlation_lines_variance: float | None = None
    """Correlation between num_lines_changed and resolve_variance."""
    
    # Summary statistics
    mean_resolve_rate: float = 0.0
    mean_resolve_variance: float = 0.0
    total_instances: int = 0
    
    # Deep variance analysis
    deep_analysis: DeepVarianceAnalysis | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'total_instances': self.total_instances,
            'mean_resolve_rate': self.mean_resolve_rate,
            'mean_resolve_variance': self.mean_resolve_variance,
            'correlation_files_resolve_rate': self.correlation_files_resolve_rate,
            'correlation_lines_resolve_rate': self.correlation_lines_resolve_rate,
            'correlation_files_variance': self.correlation_files_variance,
            'correlation_lines_variance': self.correlation_lines_variance,
        }
    
    def to_dataframe(self):
        """Convert instance metrics to pandas DataFrame.
        
        Returns:
            pandas DataFrame if pandas is available, otherwise list of dicts
        """
        if not PANDAS_AVAILABLE:
            return [m.to_dict() for m in self.instance_metrics]
        return pd.DataFrame([m.to_dict() for m in self.instance_metrics])


class VarianceAnalyzer:
    """Analyze variance across runs for the same scaffold-model pairs."""
    
    def __init__(self):
        self.metrics_extractor = MetricsExtractor()
    
    def analyze_runs(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[VarianceAnalysis]:
        """Analyze variance across runs grouped by (scaffold, base_model).
        
        Args:
            runs: List of runs to analyze
            instances: Optional dictionary mapping instance_id to Instance objects
                for difficulty metrics. If None, difficulty metrics will be None.
            
        Returns:
            List of VarianceAnalysis objects, one per scaffold-model pair
        """
        # Group runs by (scaffold, base_model)
        grouped_runs: dict[tuple[str, str], list[Run]] = defaultdict(list)
        for run in runs:
            key = (run.scaffold, run.base_model)
            grouped_runs[key].append(run)
        
        analyses = []
        
        for (scaffold, base_model), group_runs in grouped_runs.items():
            if len(group_runs) < 2:
                logger.info(
                    f"Skipping {scaffold}/{base_model}: only {len(group_runs)} run(s), "
                    "need at least 2 for variance analysis"
                )
                continue
            
            logger.info(
                f"Analyzing variance for {scaffold}/{base_model} "
                f"across {len(group_runs)} runs"
            )
            
            analysis = self._analyze_group(
                scaffold=scaffold,
                base_model=base_model,
                runs=group_runs,
                instances=instances,
            )
            analyses.append(analysis)
        
        return analyses
    
    def analyze_deep_variance(
        self,
        runs: list[Run],
    ) -> list[DeepVarianceAnalysis]:
        """Perform deep variance analysis computing mean/stddev for all metrics.
        
        Args:
            runs: List of runs to analyze
            
        Returns:
            List of DeepVarianceAnalysis objects, one per scaffold-model pair
        """
        # Group runs by (scaffold, base_model)
        grouped_runs: dict[tuple[str, str], list[Run]] = defaultdict(list)
        for run in runs:
            key = (run.scaffold, run.base_model)
            grouped_runs[key].append(run)
        
        analyses = []
        
        for (scaffold, base_model), group_runs in grouped_runs.items():
            if len(group_runs) < 2:
                logger.info(
                    f"Skipping {scaffold}/{base_model}: only {len(group_runs)} run(s), "
                    "need at least 2 for variance analysis"
                )
                continue
            
            logger.info(
                f"Performing deep variance analysis for {scaffold}/{base_model} "
                f"across {len(group_runs)} runs"
            )
            
            analysis = self._analyze_deep_group(
                scaffold=scaffold,
                base_model=base_model,
                runs=group_runs,
            )
            analyses.append(analysis)
        
        return analyses
    
    def _analyze_deep_group(
        self,
        scaffold: str,
        base_model: str,
        runs: list[Run],
    ) -> DeepVarianceAnalysis:
        """Perform deep variance analysis for a single scaffold-model group.
        
        Args:
            scaffold: Scaffold name
            base_model: Base model name
            runs: List of runs for this scaffold-model pair
            
        Returns:
            DeepVarianceAnalysis object
        """
        analysis = DeepVarianceAnalysis(
            scaffold=scaffold,
            base_model=base_model,
            num_runs=len(runs),
        )
        
        # Extract metrics for each run
        run_metrics_list: list[RunMetrics] = []
        for run in runs:
            metrics = self.metrics_extractor.extract_run_metrics(run)
            run_metrics_list.append(metrics)
        
        # Collect values for each metric
        analysis.resolve_rate.values = [m.resolve_rate for m in run_metrics_list]
        analysis.patch_rate.values = [m.patch_rate for m in run_metrics_list]
        
        analysis.total_instances.values = [m.total_instances for m in run_metrics_list]
        analysis.resolved_instances.values = [m.resolved_instances for m in run_metrics_list]
        analysis.instances_with_patch.values = [m.instances_with_patch for m in run_metrics_list]
        
        analysis.avg_tool_calls.values = [m.avg_tool_calls for m in run_metrics_list]
        analysis.median_tool_calls.values = [m.median_tool_calls for m in run_metrics_list]
        analysis.total_tool_calls.values = [m.total_tool_calls for m in run_metrics_list]
        
        analysis.avg_tokens.values = [m.avg_tokens for m in run_metrics_list]
        analysis.avg_steps.values = [m.avg_steps for m in run_metrics_list]
        
        analysis.resolved_avg_tool_calls.values = [m.resolved_avg_tool_calls for m in run_metrics_list]
        analysis.unresolved_avg_tool_calls.values = [m.unresolved_avg_tool_calls for m in run_metrics_list]
        analysis.resolved_avg_tokens.values = [m.resolved_avg_tokens for m in run_metrics_list]
        analysis.unresolved_avg_tokens.values = [m.unresolved_avg_tokens for m in run_metrics_list]
        
        # Compute token breakdowns if available
        input_tokens = []
        output_tokens = []
        for run in runs:
            if run.trajectories:
                input_tokens.append(np.mean([t.total_input_tokens for t in run.trajectories]))
                output_tokens.append(np.mean([t.total_output_tokens for t in run.trajectories]))
        
        if input_tokens:
            analysis.avg_input_tokens.values = input_tokens
        if output_tokens:
            analysis.avg_output_tokens.values = output_tokens
        
        # Aggregate tool success rates across runs
        all_tools = set()
        for metrics in run_metrics_list:
            all_tools.update(metrics.tool_success_rates.keys())
        
        for tool in all_tools:
            tool_rates = [
                metrics.tool_success_rates.get(tool, 0.0)
                for metrics in run_metrics_list
            ]
            analysis.tool_success_rates[tool] = RunMetricsStatistics(values=tool_rates)
        
        # Aggregate tool counts across runs
        all_tool_names = set()
        for metrics in run_metrics_list:
            all_tool_names.update(metrics.tool_counts.keys())
        
        for tool in all_tool_names:
            tool_counts = [
                metrics.tool_counts.get(tool, 0)
                for metrics in run_metrics_list
            ]
            analysis.tool_counts_mean[tool] = np.mean(tool_counts)
            analysis.tool_counts_stddev[tool] = np.std(tool_counts, ddof=1) if len(tool_counts) > 1 else 0.0
        
        # Compute statistics for all metrics
        for field_name in [
            'resolve_rate', 'patch_rate', 'total_instances', 'resolved_instances',
            'instances_with_patch', 'avg_tool_calls', 'median_tool_calls', 'total_tool_calls',
            'avg_tokens', 'avg_input_tokens', 'avg_output_tokens', 'avg_steps',
            'resolved_avg_tool_calls', 'unresolved_avg_tool_calls',
            'resolved_avg_tokens', 'unresolved_avg_tokens',
        ]:
            stats = getattr(analysis, field_name)
            stats.__post_init__()
        
        return analysis
    
    def _analyze_group(
        self,
        scaffold: str,
        base_model: str,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> VarianceAnalysis:
        """Analyze variance for a single scaffold-model group.
        
        Args:
            scaffold: Scaffold name
            base_model: Base model name
            runs: List of runs for this scaffold-model pair
            instances: Optional instances dictionary for difficulty metrics
            
        Returns:
            VarianceAnalysis object
        """
        analysis = VarianceAnalysis(
            scaffold=scaffold,
            base_model=base_model,
        )
        
        # Group trajectories by instance_id across all runs
        instance_outcomes: dict[str, dict[str, list[bool]]] = defaultdict(
            lambda: {'resolved': [], 'has_patch': []}
        )
        
        for run in runs:
            for traj in run.trajectories:
                instance_id = traj.instance_id
                instance_outcomes[instance_id]['resolved'].append(
                    traj.resolved is True
                )
                instance_outcomes[instance_id]['has_patch'].append(
                    traj.has_patch
                )
        
        # Create InstanceVarianceMetrics for each instance
        for instance_id, outcomes in instance_outcomes.items():
            # Get difficulty metrics from instances if available
            instance = instances.get(instance_id) if instances else None
            
            metrics = InstanceVarianceMetrics(
                instance_id=instance_id,
                scaffold=scaffold,
                base_model=base_model,
                resolved_outcomes=outcomes['resolved'],
                patch_outcomes=outcomes['has_patch'],
            )
            
            # Add difficulty metrics if available
            if instance:
                metrics.num_files_changed = instance.num_files_changed
                metrics.num_lines_changed = instance.num_lines_changed
                metrics.num_additions = instance.num_additions
                metrics.num_deletions = instance.num_deletions
            
            analysis.instance_metrics.append(metrics)
        
        # Compute summary statistics
        analysis.total_instances = len(analysis.instance_metrics)
        if analysis.instance_metrics:
            analysis.mean_resolve_rate = np.mean([
                m.resolve_rate for m in analysis.instance_metrics
            ])
            analysis.mean_resolve_variance = np.mean([
                m.resolve_variance for m in analysis.instance_metrics
            ])
        
        # Compute correlations if difficulty metrics are available
        if not PANDAS_AVAILABLE:
            logger.warning("pandas not available, skipping correlation calculations")
        else:
            df = analysis.to_dataframe()
            
            if instances and len(df) > 0:
                # Filter out instances without difficulty metrics
                df_with_difficulty = df[
                    df['num_files_changed'].notna() & df['num_lines_changed'].notna()
                ]
                
                if len(df_with_difficulty) > 1:
                    # Correlation: files vs resolve_rate
                    if df_with_difficulty['num_files_changed'].std() > 0:
                        analysis.correlation_files_resolve_rate = df_with_difficulty[
                            ['num_files_changed', 'resolve_rate']
                        ].corr().iloc[0, 1]
                    
                    # Correlation: lines vs resolve_rate
                    if df_with_difficulty['num_lines_changed'].std() > 0:
                        analysis.correlation_lines_resolve_rate = df_with_difficulty[
                            ['num_lines_changed', 'resolve_rate']
                        ].corr().iloc[0, 1]
                    
                    # Correlation: files vs variance
                    if df_with_difficulty['num_files_changed'].std() > 0:
                        analysis.correlation_files_variance = df_with_difficulty[
                            ['num_files_changed', 'resolve_variance']
                        ].corr().iloc[0, 1]
                    
                    # Correlation: lines vs variance
                    if df_with_difficulty['num_lines_changed'].std() > 0:
                        analysis.correlation_lines_variance = df_with_difficulty[
                            ['num_lines_changed', 'resolve_variance']
                        ].corr().iloc[0, 1]
        
        # Add deep variance analysis
        analysis.deep_analysis = self._analyze_deep_group(
            scaffold=scaffold,
            base_model=base_model,
            runs=runs,
        )
        
        return analysis

