"""Trajectory dynamics analysis: temporal patterns, localization, recovery, and sequences."""

import re
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models import Run, Trajectory, Instance

logger = logging.getLogger(__name__)


# Tools that modify files
EDIT_TOOLS = {
    'str_replace_editor', 'apply_patch', 'file_editor', 'edit', 'create',
    'str_replace', 'insert', 'write_file',
}

# Tools that read/explore files
EXPLORE_TOOLS = {
    'view', 'read_file', 'cat', 'head', 'tail', 'open_file',
}

# Tools that search
SEARCH_TOOLS = {
    'search', 'search_file', 'search_dir', 'find_file', 'grep', 'rg', 'find',
}

# Shell tools
SHELL_TOOLS = {
    'shell', 'bash', 'execute_bash', 'run_command', 'terminal',
}


def extract_files_from_patch(patch: str) -> set[str]:
    """Extract file paths from a unified diff patch.
    
    Args:
        patch: Unified diff patch string
        
    Returns:
        Set of file paths
    """
    if not patch:
        return set()
    
    # Match: --- a/path/to/file.py or +++ b/path/to/file.py
    file_pattern = re.compile(r'^(?:---|\+\+\+)\s+[ab]/(.+?)(?:\t|$)', re.MULTILINE)
    
    files = set()
    for match in file_pattern.finditer(patch):
        file_path = match.group(1).strip()
        if file_path and file_path != '/dev/null':
            files.add(file_path)
    
    return files


def extract_files_touched(traj: Trajectory) -> set[str]:
    """Extract all files the agent touched (read, searched, or edited).
    
    Args:
        traj: Trajectory to analyze
        
    Returns:
        Set of file paths
    """
    files = set()
    
    for step in traj.steps:
        for tc in step.tool_calls:
            # Extract from arguments
            for key in ['path', 'file_path', 'file', 'filename']:
                if key in tc.arguments:
                    path = tc.arguments[key]
                    if isinstance(path, str) and path:
                        # Clean up path
                        path = path.lstrip('./')
                        files.add(path)
            
            # Extract from shell commands (grep, cat, etc.)
            if tc.shell_command in ('grep', 'rg', 'cat', 'head', 'tail', 'less', 'vim', 'nano'):
                cmd = tc.arguments.get('cmd', '') or tc.arguments.get('command', '')
                # Try to extract file paths from command
                parts = cmd.split()
                for part in parts:
                    if '.' in part and not part.startswith('-'):
                        # Looks like a file path
                        path = part.lstrip('./')
                        if not path.startswith('-'):
                            files.add(path)
    
    return files


def extract_files_edited(traj: Trajectory) -> set[str]:
    """Extract files the agent attempted to edit.
    
    Args:
        traj: Trajectory to analyze
        
    Returns:
        Set of file paths
    """
    files = set()
    
    for step in traj.steps:
        for tc in step.tool_calls:
            if tc.name in EDIT_TOOLS:
                for key in ['path', 'file_path', 'file', 'filename']:
                    if key in tc.arguments:
                        path = tc.arguments[key]
                        if isinstance(path, str) and path:
                            path = path.lstrip('./')
                            files.add(path)
    
    return files


@dataclass
class LocalizationMetrics:
    """Metrics for how well the agent localized the problem."""
    
    instance_id: str
    
    # Files in ground truth
    oracle_files: set[str] = field(default_factory=set)
    
    # Files agent interacted with
    files_touched: set[str] = field(default_factory=set)
    files_edited: set[str] = field(default_factory=set)
    
    # Computed metrics
    touch_precision: float = 0.0  # |touched ∩ oracle| / |touched|
    touch_recall: float = 0.0     # |touched ∩ oracle| / |oracle|
    edit_precision: float = 0.0   # |edited ∩ oracle| / |edited|
    edit_recall: float = 0.0      # |edited ∩ oracle| / |oracle|
    
    # Detailed breakdown
    correct_files_touched: set[str] = field(default_factory=set)
    false_positives: set[str] = field(default_factory=set)  # touched but not in oracle
    missed_files: set[str] = field(default_factory=set)     # in oracle but not touched
    
    def __post_init__(self):
        """Compute metrics from file sets."""
        if self.oracle_files and self.files_touched:
            self.correct_files_touched = self.oracle_files & self.files_touched
            self.false_positives = self.files_touched - self.oracle_files
            self.missed_files = self.oracle_files - self.files_touched
            
            self.touch_precision = len(self.correct_files_touched) / len(self.files_touched)
            self.touch_recall = len(self.correct_files_touched) / len(self.oracle_files)
        
        if self.oracle_files and self.files_edited:
            correct_edits = self.oracle_files & self.files_edited
            self.edit_precision = len(correct_edits) / len(self.files_edited)
            self.edit_recall = len(correct_edits) / len(self.oracle_files)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'oracle_files_count': len(self.oracle_files),
            'files_touched_count': len(self.files_touched),
            'files_edited_count': len(self.files_edited),
            'touch_precision': self.touch_precision,
            'touch_recall': self.touch_recall,
            'edit_precision': self.edit_precision,
            'edit_recall': self.edit_recall,
            'correct_files_count': len(self.correct_files_touched),
            'false_positives_count': len(self.false_positives),
            'missed_files_count': len(self.missed_files),
        }


@dataclass
class PhaseMetrics:
    """Metrics about trajectory phases (exploration → editing → verification)."""
    
    total_steps: int = 0
    
    # Phase step counts
    exploration_steps: int = 0   # Steps before first edit
    editing_steps: int = 0       # Steps with edits
    verification_steps: int = 0  # Steps after last edit
    
    # Tool type counts
    search_tool_calls: int = 0
    explore_tool_calls: int = 0
    edit_tool_calls: int = 0
    shell_tool_calls: int = 0
    
    # First edit timing
    steps_to_first_edit: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_steps': self.total_steps,
            'exploration_steps': self.exploration_steps,
            'editing_steps': self.editing_steps,
            'verification_steps': self.verification_steps,
            'search_tool_calls': self.search_tool_calls,
            'explore_tool_calls': self.explore_tool_calls,
            'edit_tool_calls': self.edit_tool_calls,
            'shell_tool_calls': self.shell_tool_calls,
            'steps_to_first_edit': self.steps_to_first_edit,
        }


@dataclass
class ErrorRecoveryMetrics:
    """Metrics about error recovery behavior."""
    
    total_tool_calls: int = 0
    failed_tool_calls: int = 0
    error_rate: float = 0.0
    
    # Recovery metrics
    errors_followed_by_success: int = 0  # Error then eventually resolved
    max_consecutive_errors: int = 0
    
    # Error patterns
    error_positions: list[int] = field(default_factory=list)  # Step indices with errors
    recovery_rate: float = 0.0  # Fraction of errors followed by recovery
    
    # Per-tool error rates
    tool_error_counts: dict[str, int] = field(default_factory=dict)
    tool_total_counts: dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_tool_calls': self.total_tool_calls,
            'failed_tool_calls': self.failed_tool_calls,
            'error_rate': self.error_rate,
            'max_consecutive_errors': self.max_consecutive_errors,
            'recovery_rate': self.recovery_rate,
        }


@dataclass
class SequencePatterns:
    """Tool sequence pattern analysis."""
    
    # N-gram patterns
    success_bigrams: Counter = field(default_factory=Counter)
    failure_bigrams: Counter = field(default_factory=Counter)
    success_trigrams: Counter = field(default_factory=Counter)
    failure_trigrams: Counter = field(default_factory=Counter)
    
    # Distinctive patterns
    success_only_patterns: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    failure_only_patterns: list[tuple[tuple[str, ...], int]] = field(default_factory=list)
    
    # First N tools
    success_starts: Counter = field(default_factory=Counter)  # First 3 tools
    failure_starts: Counter = field(default_factory=Counter)
    
    # Last N tools  
    success_ends: Counter = field(default_factory=Counter)  # Last 3 tools
    failure_ends: Counter = field(default_factory=Counter)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'top_success_bigrams': self.success_bigrams.most_common(20),
            'top_failure_bigrams': self.failure_bigrams.most_common(20),
            'top_success_trigrams': self.success_trigrams.most_common(15),
            'top_failure_trigrams': self.failure_trigrams.most_common(15),
            'success_only_patterns': self.success_only_patterns[:10],
            'failure_only_patterns': self.failure_only_patterns[:10],
            'top_success_starts': self.success_starts.most_common(10),
            'top_failure_starts': self.failure_starts.most_common(10),
            'top_success_ends': self.success_ends.most_common(10),
            'top_failure_ends': self.failure_ends.most_common(10),
        }


@dataclass
class TrajectoryDynamicsAnalysis:
    """Complete dynamics analysis for a run."""
    
    run_name: str
    scaffold: str
    base_model: str
    
    # Aggregated localization metrics
    mean_touch_precision: float = 0.0
    mean_touch_recall: float = 0.0
    mean_edit_precision: float = 0.0
    mean_edit_recall: float = 0.0
    
    # Aggregated phase metrics
    mean_exploration_steps: float = 0.0
    mean_editing_steps: float = 0.0
    mean_verification_steps: float = 0.0
    mean_steps_to_first_edit: float = 0.0
    
    # Aggregated error metrics
    mean_error_rate: float = 0.0
    mean_max_consecutive_errors: float = 0.0
    overall_recovery_rate: float = 0.0  # Trajectories with errors that still resolved
    
    # Sequence patterns
    sequence_patterns: SequencePatterns = field(default_factory=SequencePatterns)
    
    # Per-instance metrics
    localization_metrics: list[LocalizationMetrics] = field(default_factory=list)
    phase_metrics: list[PhaseMetrics] = field(default_factory=list)
    error_metrics: list[ErrorRecoveryMetrics] = field(default_factory=list)
    
    # Resolved vs unresolved comparison
    resolved_mean_touch_recall: float = 0.0
    unresolved_mean_touch_recall: float = 0.0
    resolved_mean_steps_to_first_edit: float = 0.0
    unresolved_mean_steps_to_first_edit: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_name': self.run_name,
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'mean_touch_precision': self.mean_touch_precision,
            'mean_touch_recall': self.mean_touch_recall,
            'mean_edit_precision': self.mean_edit_precision,
            'mean_edit_recall': self.mean_edit_recall,
            'mean_exploration_steps': self.mean_exploration_steps,
            'mean_editing_steps': self.mean_editing_steps,
            'mean_verification_steps': self.mean_verification_steps,
            'mean_steps_to_first_edit': self.mean_steps_to_first_edit,
            'mean_error_rate': self.mean_error_rate,
            'overall_recovery_rate': self.overall_recovery_rate,
            'resolved_mean_touch_recall': self.resolved_mean_touch_recall,
            'unresolved_mean_touch_recall': self.unresolved_mean_touch_recall,
            'sequence_patterns': self.sequence_patterns.to_dict(),
        }


class TrajectoryDynamicsAnalyzer:
    """Analyze trajectory dynamics: localization, phases, recovery, and sequences."""
    
    def analyze_run(
        self,
        run: Run,
        instances: dict[str, Instance] | None = None,
    ) -> TrajectoryDynamicsAnalysis:
        """Perform complete dynamics analysis on a run.
        
        Args:
            run: Run to analyze
            instances: Optional instances dict for ground truth patches
            
        Returns:
            TrajectoryDynamicsAnalysis with all metrics
        """
        analysis = TrajectoryDynamicsAnalysis(
            run_name=run.name,
            scaffold=run.scaffold,
            base_model=run.base_model,
        )
        
        # Analyze each trajectory
        for traj in run.trajectories:
            instance = instances.get(traj.instance_id) if instances else None
            
            # Localization analysis
            loc_metrics = self._analyze_localization(traj, instance)
            analysis.localization_metrics.append(loc_metrics)
            
            # Phase analysis
            phase_metrics = self._analyze_phases(traj)
            analysis.phase_metrics.append(phase_metrics)
            
            # Error recovery analysis
            error_metrics = self._analyze_errors(traj)
            analysis.error_metrics.append(error_metrics)
        
        # Sequence pattern analysis
        analysis.sequence_patterns = self._analyze_sequences(run)
        
        # Compute aggregated metrics
        self._compute_aggregates(analysis, run)
        
        return analysis
    
    def _analyze_localization(
        self,
        traj: Trajectory,
        instance: Instance | None,
    ) -> LocalizationMetrics:
        """Analyze localization accuracy for a trajectory."""
        oracle_files = set()
        if instance and instance.patch:
            oracle_files = extract_files_from_patch(instance.patch)
        
        files_touched = extract_files_touched(traj)
        files_edited = extract_files_edited(traj)
        
        return LocalizationMetrics(
            instance_id=traj.instance_id,
            oracle_files=oracle_files,
            files_touched=files_touched,
            files_edited=files_edited,
        )
    
    def _analyze_phases(self, traj: Trajectory) -> PhaseMetrics:
        """Analyze trajectory phases."""
        metrics = PhaseMetrics(total_steps=len(traj.steps))
        
        first_edit_step = None
        last_edit_step = None
        
        for step_idx, step in enumerate(traj.steps):
            has_edit = False
            
            for tc in step.tool_calls:
                # Count tool types
                if tc.name in SEARCH_TOOLS or tc.shell_command in ('grep', 'rg', 'find', 'ag'):
                    metrics.search_tool_calls += 1
                elif tc.name in EXPLORE_TOOLS or tc.shell_command in ('cat', 'head', 'tail', 'less'):
                    metrics.explore_tool_calls += 1
                elif tc.name in EDIT_TOOLS:
                    metrics.edit_tool_calls += 1
                    has_edit = True
                elif tc.name in SHELL_TOOLS:
                    metrics.shell_tool_calls += 1
            
            if has_edit:
                if first_edit_step is None:
                    first_edit_step = step_idx
                last_edit_step = step_idx
        
        # Compute phase step counts
        if first_edit_step is not None:
            metrics.steps_to_first_edit = first_edit_step
            metrics.exploration_steps = first_edit_step
            metrics.editing_steps = last_edit_step - first_edit_step + 1
            metrics.verification_steps = len(traj.steps) - last_edit_step - 1
        else:
            # No edits - all exploration
            metrics.exploration_steps = len(traj.steps)
        
        return metrics
    
    def _analyze_errors(self, traj: Trajectory) -> ErrorRecoveryMetrics:
        """Analyze error recovery behavior."""
        metrics = ErrorRecoveryMetrics()
        
        consecutive_errors = 0
        max_consecutive = 0
        
        for step_idx, step in enumerate(traj.steps):
            for tc in step.tool_calls:
                metrics.total_tool_calls += 1
                
                # Track per-tool counts
                metrics.tool_total_counts[tc.name] = metrics.tool_total_counts.get(tc.name, 0) + 1
                
                if not tc.success:
                    metrics.failed_tool_calls += 1
                    metrics.error_positions.append(step_idx)
                    consecutive_errors += 1
                    max_consecutive = max(max_consecutive, consecutive_errors)
                    
                    # Track per-tool errors
                    metrics.tool_error_counts[tc.name] = metrics.tool_error_counts.get(tc.name, 0) + 1
                else:
                    consecutive_errors = 0
        
        metrics.max_consecutive_errors = max_consecutive
        
        if metrics.total_tool_calls > 0:
            metrics.error_rate = metrics.failed_tool_calls / metrics.total_tool_calls
        
        return metrics
    
    def _analyze_sequences(self, run: Run) -> SequencePatterns:
        """Analyze tool call sequence patterns."""
        patterns = SequencePatterns()
        
        for traj in run.trajectories:
            # Extract tool sequence
            tools = []
            for step in traj.steps:
                for tc in step.tool_calls:
                    tools.append(tc.name)
            
            if not tools:
                continue
            
            # Compute n-grams
            bigrams = list(zip(tools[:-1], tools[1:]))
            trigrams = list(zip(tools[:-2], tools[1:-1], tools[2:]))
            
            # Track starts and ends
            start = tuple(tools[:3]) if len(tools) >= 3 else tuple(tools)
            end = tuple(tools[-3:]) if len(tools) >= 3 else tuple(tools)
            
            if traj.resolved:
                patterns.success_bigrams.update(bigrams)
                patterns.success_trigrams.update(trigrams)
                patterns.success_starts[start] += 1
                patterns.success_ends[end] += 1
            else:
                patterns.failure_bigrams.update(bigrams)
                patterns.failure_trigrams.update(trigrams)
                patterns.failure_starts[start] += 1
                patterns.failure_ends[end] += 1
        
        # Find distinctive patterns (appear only in success or failure)
        success_only = set(patterns.success_bigrams.keys()) - set(patterns.failure_bigrams.keys())
        failure_only = set(patterns.failure_bigrams.keys()) - set(patterns.success_bigrams.keys())
        
        patterns.success_only_patterns = [
            (p, patterns.success_bigrams[p]) for p in success_only
        ]
        patterns.success_only_patterns.sort(key=lambda x: x[1], reverse=True)
        
        patterns.failure_only_patterns = [
            (p, patterns.failure_bigrams[p]) for p in failure_only
        ]
        patterns.failure_only_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return patterns
    
    def _compute_aggregates(
        self,
        analysis: TrajectoryDynamicsAnalysis,
        run: Run,
    ) -> None:
        """Compute aggregated metrics."""
        # Localization aggregates
        touch_precisions = [m.touch_precision for m in analysis.localization_metrics if m.oracle_files]
        touch_recalls = [m.touch_recall for m in analysis.localization_metrics if m.oracle_files]
        edit_precisions = [m.edit_precision for m in analysis.localization_metrics if m.oracle_files]
        edit_recalls = [m.edit_recall for m in analysis.localization_metrics if m.oracle_files]
        
        if touch_precisions:
            analysis.mean_touch_precision = np.mean(touch_precisions)
        if touch_recalls:
            analysis.mean_touch_recall = np.mean(touch_recalls)
        if edit_precisions:
            analysis.mean_edit_precision = np.mean(edit_precisions)
        if edit_recalls:
            analysis.mean_edit_recall = np.mean(edit_recalls)
        
        # Phase aggregates
        exploration = [m.exploration_steps for m in analysis.phase_metrics]
        editing = [m.editing_steps for m in analysis.phase_metrics]
        verification = [m.verification_steps for m in analysis.phase_metrics]
        first_edits = [m.steps_to_first_edit for m in analysis.phase_metrics if m.steps_to_first_edit is not None]
        
        if exploration:
            analysis.mean_exploration_steps = np.mean(exploration)
        if editing:
            analysis.mean_editing_steps = np.mean(editing)
        if verification:
            analysis.mean_verification_steps = np.mean(verification)
        if first_edits:
            analysis.mean_steps_to_first_edit = np.mean(first_edits)
        
        # Error aggregates
        error_rates = [m.error_rate for m in analysis.error_metrics]
        max_consecutive = [m.max_consecutive_errors for m in analysis.error_metrics]
        
        if error_rates:
            analysis.mean_error_rate = np.mean(error_rates)
        if max_consecutive:
            analysis.mean_max_consecutive_errors = np.mean(max_consecutive)
        
        # Recovery rate: trajectories with errors that still resolved
        trajs_with_errors = 0
        resolved_with_errors = 0
        for traj, err_metrics in zip(run.trajectories, analysis.error_metrics):
            if err_metrics.failed_tool_calls > 0:
                trajs_with_errors += 1
                if traj.resolved:
                    resolved_with_errors += 1
        
        if trajs_with_errors > 0:
            analysis.overall_recovery_rate = resolved_with_errors / trajs_with_errors
        
        # Resolved vs unresolved comparison
        resolved_recalls = []
        unresolved_recalls = []
        resolved_first_edits = []
        unresolved_first_edits = []
        
        for traj, loc_m, phase_m in zip(
            run.trajectories,
            analysis.localization_metrics,
            analysis.phase_metrics
        ):
            if loc_m.oracle_files:
                if traj.resolved:
                    resolved_recalls.append(loc_m.touch_recall)
                else:
                    unresolved_recalls.append(loc_m.touch_recall)
            
            if phase_m.steps_to_first_edit is not None:
                if traj.resolved:
                    resolved_first_edits.append(phase_m.steps_to_first_edit)
                else:
                    unresolved_first_edits.append(phase_m.steps_to_first_edit)
        
        if resolved_recalls:
            analysis.resolved_mean_touch_recall = np.mean(resolved_recalls)
        if unresolved_recalls:
            analysis.unresolved_mean_touch_recall = np.mean(unresolved_recalls)
        if resolved_first_edits:
            analysis.resolved_mean_steps_to_first_edit = np.mean(resolved_first_edits)
        if unresolved_first_edits:
            analysis.unresolved_mean_steps_to_first_edit = np.mean(unresolved_first_edits)
    
    def analyze_runs(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[TrajectoryDynamicsAnalysis]:
        """Analyze multiple runs.
        
        Args:
            runs: List of runs to analyze
            instances: Optional instances dict for ground truth
            
        Returns:
            List of TrajectoryDynamicsAnalysis objects
        """
        analyses = []
        for run in runs:
            logger.info(f"Analyzing dynamics for run: {run.name}")
            analysis = self.analyze_run(run, instances)
            analyses.append(analysis)
        
        return analyses
