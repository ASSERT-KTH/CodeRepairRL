"""Plotting functions for trajectory analysis visualization."""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..models import Run, Instance
from ..analysis import MetricsExtractor, RunMetrics, RunComparator
from ..analysis.transfer import (
    TransferAnalyzer, 
    DeepTransferAnalysis, 
    extract_error_pattern, 
    extract_raw_error_prefix,
    format_tool_call_signature,
)
from ..analysis.variance import (
    VarianceAnalyzer, VarianceAnalysis, DeepVarianceAnalysis, RunMetricsStatistics
)
from ..analysis.dynamics import (
    TrajectoryDynamicsAnalyzer, TrajectoryDynamicsAnalysis, SequencePatterns
)
from ..analysis.failure import (
    FailureTaxonomyAnalyzer, FailureTaxonomy, FailureCategory
)

logger = logging.getLogger(__name__)


class TrajectoryPlotter:
    """Generate plots for trajectory analysis."""
    
    def __init__(self, output_dir: Path | str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_extractor = MetricsExtractor()
        self.comparator = RunComparator()
    
    def plot_all(
        self,
        runs: list[Run],
        plots: list[str] | None = None,
        model_to_trained_scaffold: dict[str, str] | None = None,
        instances: dict[str, Instance] | None = None,
    ) -> list[Path]:
        """Generate all requested plots.
        
        Args:
            runs: List of runs to analyze
            plots: List of plot types to generate. If None, generate all.
            model_to_trained_scaffold: Mapping of model names to their training scaffold
                (needed for transfer_* plots)
            
        Returns:
            List of generated plot file paths
        """
        all_plots = [
            'tool_distribution',
            'shell_command_distribution',
            'success_rates',
            'token_analysis',
            'error_patterns',
            'failed_tool_calls',
            'error_analysis',            # comprehensive error analysis
            'comparison',
            'transfer_analysis',          # high-level transfer summary
            'transfer_mcnemar',           # instance-level transfer (McNemar)
            'transfer_error_modes',       # error mode comparison
            'transfer_vocabulary',        # tool vocabulary alignment
            'transfer_error_patterns',    # error pattern deltas
            'variance_vs_difficulty',     # variance analysis vs difficulty metrics
            'deep_variance',              # deep variance analysis with mean/stddev for all metrics
            'trajectory_dynamics',        # localization, phases, error recovery
            'sequence_patterns',          # tool call sequence analysis
            'failure_taxonomy',           # categorization of why trajectories fail
        ]
        
        if plots is None:
            plots = all_plots
        
        generated: list[Path] = []
        
        for run in runs:
            if 'tool_distribution' in plots:
                path = self.plot_tool_distribution(run)
                if path:
                    generated.append(path)
            
            if 'shell_command_distribution' in plots:
                path = self.plot_shell_command_distribution(run)
                if path:
                    generated.append(path)
            
            if 'success_rates' in plots:
                path = self.plot_success_rates(run)
                if path:
                    generated.append(path)
            
            if 'token_analysis' in plots:
                path = self.plot_token_analysis(run)
                if path:
                    generated.append(path)
            
            if 'error_patterns' in plots:
                path = self.plot_error_patterns(run)
                if path:
                    generated.append(path)
            
            if 'failed_tool_calls' in plots:
                path = self.plot_failed_tool_calls(run)
                if path:
                    generated.append(path)
            
            if 'error_analysis' in plots:
                path = self.plot_error_analysis(run)
                if path:
                    generated.append(path)
        
        if len(runs) > 1:
            if 'comparison' in plots:
                path = self.plot_comparison(runs)
                if path:
                    generated.append(path)
            
            if 'transfer_analysis' in plots:
                path = self.plot_transfer_analysis(runs, model_to_trained_scaffold)
                if path:
                    generated.append(path)
            
            # Transfer sub-plots (per transfer pair)
            if 'transfer_mcnemar' in plots:
                generated.extend(
                    self.plot_transfer_mcnemar_all(runs, model_to_trained_scaffold)
                )
            if 'transfer_error_modes' in plots:
                generated.extend(
                    self.plot_transfer_error_modes_all(runs, model_to_trained_scaffold)
                )
            if 'transfer_vocabulary' in plots:
                generated.extend(
                    self.plot_transfer_vocabulary_all(runs, model_to_trained_scaffold)
                )
            if 'transfer_error_patterns' in plots:
                generated.extend(
                    self.plot_transfer_error_patterns_all(runs, model_to_trained_scaffold)
                )
            
            if 'variance_vs_difficulty' in plots:
                generated.extend(
                    self.plot_variance_vs_difficulty_all(runs, instances)
                )
            
            if 'deep_variance' in plots:
                generated.extend(
                    self.plot_deep_variance_all(runs)
                )
            
            if 'trajectory_dynamics' in plots:
                generated.extend(
                    self.plot_trajectory_dynamics_all(runs, instances)
                )
            
            if 'sequence_patterns' in plots:
                generated.extend(
                    self.plot_sequence_patterns_all(runs)
                )
            
            if 'failure_taxonomy' in plots:
                generated.extend(
                    self.plot_failure_taxonomy_all(runs, instances)
                )
        
        return generated
    
    def plot_tool_distribution(self, run: Run) -> Path | None:
        """Plot distribution of tools used in a run."""
        tool_counts = run.get_total_tool_counts()
        if not tool_counts:
            logger.warning(f"No tool data found for {run.name}")
            return None
        
        # Sort by count
        sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        tools, counts = zip(*sorted_tools)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(tools)), counts, color='steelblue', alpha=0.7)
        plt.xlabel('Tool', fontsize=12)
        plt.ylabel('Total Count', fontsize=12)
        plt.title(f'Tool Distribution - {run.name}', fontsize=14, fontweight='bold')
        plt.xticks(range(len(tools)), tools, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_tool_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved tool distribution plot to {output_path}")
        return output_path
    
    def plot_shell_command_distribution(self, run: Run) -> Path | None:
        """Plot distribution of shell commands used."""
        shell_counts = run.get_total_shell_command_counts()
        if not shell_counts:
            logger.warning(f"No shell command data found for {run.name}")
            return None
        
        # Sort by count and take top 20
        sorted_cmds = sorted(shell_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        commands, counts = zip(*sorted_cmds)
        
        plt.figure(figsize=(14, 7))
        bars = plt.bar(range(len(commands)), counts, color='coral', alpha=0.7)
        plt.xlabel('Shell Command', fontsize=12)
        plt.ylabel('Total Count', fontsize=12)
        plt.title(f'Shell Command Distribution - {run.name}', fontsize=14, fontweight='bold')
        plt.xticks(range(len(commands)), commands, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_shell_command_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved shell command distribution plot to {output_path}")
        return output_path
    
    def plot_success_rates(self, run: Run) -> Path | None:
        """Plot success rates and metrics for a run."""
        metrics = self.metrics_extractor.extract_run_metrics(run)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Success Rates and Metrics - {run.name}', fontsize=16, fontweight='bold')
        
        # 1. Resolution rate pie chart
        resolved = metrics.resolved_instances
        unresolved = metrics.total_instances - resolved
        
        if metrics.total_instances > 0:
            axes[0, 0].pie(
                [resolved, unresolved],
                labels=[f'Resolved\n({resolved})', f'Unresolved\n({unresolved})'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['#2ecc71', '#e74c3c']
            )
            axes[0, 0].set_title('Resolution Rate')
        
        # 2. Tool success rates
        if metrics.tool_success_rates:
            sorted_rates = sorted(metrics.tool_success_rates.items(), key=lambda x: x[1], reverse=True)
            tools, rates = zip(*sorted_rates[:10])  # Top 10
            
            bars = axes[0, 1].barh(range(len(tools)), rates, color='#3498db', alpha=0.7)
            axes[0, 1].set_yticks(range(len(tools)))
            axes[0, 1].set_yticklabels(tools)
            axes[0, 1].set_xlabel('Success Rate')
            axes[0, 1].set_title('Tool Success Rates')
            axes[0, 1].set_xlim(0, 1.1)
            axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Tool usage distribution
        if metrics.tool_calls_per_traj:
            axes[1, 0].hist(metrics.tool_calls_per_traj, bins=30, color='#f39c12', alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(metrics.avg_tool_calls, color='red', linestyle='--',
                             label=f'Mean: {metrics.avg_tool_calls:.1f}')
            axes[1, 0].axvline(metrics.median_tool_calls, color='blue', linestyle='--',
                             label=f'Median: {metrics.median_tool_calls:.1f}')
            axes[1, 0].set_xlabel('Total Tool Calls per Instance')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Tool Usage Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Resolved vs unresolved comparison
        if metrics.resolved_avg_tool_calls > 0 or metrics.unresolved_avg_tool_calls > 0:
            categories = ['Resolved', 'Unresolved']
            tool_values = [metrics.resolved_avg_tool_calls, metrics.unresolved_avg_tool_calls]
            colors = ['#2ecc71', '#e74c3c']
            
            bars = axes[1, 1].bar(categories, tool_values, color=colors, alpha=0.7)
            axes[1, 1].set_ylabel('Average Tool Calls')
            axes[1, 1].set_title('Avg Tool Calls: Resolved vs Unresolved')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_success_rates.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved success rates plot to {output_path}")
        return output_path
    
    def plot_token_analysis(self, run: Run) -> Path | None:
        """Plot token usage analysis."""
        metrics = self.metrics_extractor.extract_run_metrics(run)
        
        if not metrics.tokens_per_traj:
            logger.warning(f"No token data found for {run.name}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Token Usage Analysis - {run.name}', fontsize=14, fontweight='bold')
        
        # 1. Token usage by resolution status (stacked bar)
        resolved_trajs = run.get_resolved_trajectories()
        unresolved_trajs = run.get_unresolved_trajectories()
        
        if resolved_trajs or unresolved_trajs:
            # Calculate average token breakdowns
            # Input tokens breakdown: cache_read (bottom), cache_creation (middle), regular input (top)
            resolved_cache_read = np.mean([t.total_cache_read_input_tokens for t in resolved_trajs]) if resolved_trajs else 0
            resolved_cache_creation = np.mean([t.total_cache_creation_input_tokens for t in resolved_trajs]) if resolved_trajs else 0
            resolved_input = np.mean([t.total_input_tokens for t in resolved_trajs]) if resolved_trajs else 0
            resolved_output = np.mean([t.total_output_tokens for t in resolved_trajs]) if resolved_trajs else 0
            
            unresolved_cache_read = np.mean([t.total_cache_read_input_tokens for t in unresolved_trajs]) if unresolved_trajs else 0
            unresolved_cache_creation = np.mean([t.total_cache_creation_input_tokens for t in unresolved_trajs]) if unresolved_trajs else 0
            unresolved_input = np.mean([t.total_input_tokens for t in unresolved_trajs]) if unresolved_trajs else 0
            unresolved_output = np.mean([t.total_output_tokens for t in unresolved_trajs]) if unresolved_trajs else 0
            
            categories = []
            cache_read_vals = []
            cache_creation_vals = []
            input_vals = []
            output_vals = []
            
            if resolved_trajs:
                categories.append(f'Resolved\n(n={len(resolved_trajs)})')
                cache_read_vals.append(resolved_cache_read)
                cache_creation_vals.append(resolved_cache_creation)
                input_vals.append(resolved_input)
                output_vals.append(resolved_output)
            
            if unresolved_trajs:
                categories.append(f'Unresolved\n(n={len(unresolved_trajs)})')
                cache_read_vals.append(unresolved_cache_read)
                cache_creation_vals.append(unresolved_cache_creation)
                input_vals.append(unresolved_input)
                output_vals.append(unresolved_output)
            
            if categories:
                x = np.arange(len(categories))
                width = 0.6
                
                # Stack input tokens: cache_read (bottom), cache_creation (middle), regular input (top)
                bottom = np.zeros(len(categories))
                if any(cache_read_vals):
                    axes[0, 0].bar(x, cache_read_vals, width, bottom=bottom, 
                                  label='Cache Read Input Tokens', color='#9b59b6', alpha=0.7)
                    bottom += np.array(cache_read_vals)
                
                if any(cache_creation_vals):
                    axes[0, 0].bar(x, cache_creation_vals, width, bottom=bottom,
                                  label='Cache Creation Input Tokens', color='#f39c12', alpha=0.7)
                    bottom += np.array(cache_creation_vals)
                
                axes[0, 0].bar(x, input_vals, width, bottom=bottom, 
                              label='Input Tokens (after cache)', color='#3498db', alpha=0.7)
                bottom += np.array(input_vals)
                
                # Output tokens on top of all input
                axes[0, 0].bar(x, output_vals, width, bottom=bottom,
                              label='Output Tokens', color='#e74c3c', alpha=0.7)
                
                axes[0, 0].set_xticks(x)
                axes[0, 0].set_xticklabels(categories)
                axes[0, 0].set_ylabel('Average Token Usage')
                axes[0, 0].set_title('Average Token Usage by Resolution Status')
                axes[0, 0].legend(fontsize=8)
                axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Token usage distribution (boxplot with total tokens)
        tokens_resolved = [t.total_tokens for t in resolved_trajs]
        tokens_unresolved = [t.total_tokens for t in unresolved_trajs]
        
        if tokens_resolved or tokens_unresolved:
            data = []
            labels = []
            if tokens_resolved:
                data.append(tokens_resolved)
                labels.append(f'Resolved\n(n={len(tokens_resolved)})')
            if tokens_unresolved:
                data.append(tokens_unresolved)
                labels.append(f'Unresolved\n(n={len(tokens_unresolved)})')
            
            bp = axes[0, 1].boxplot(data, labels=labels, patch_artist=True)
            if len(bp['boxes']) > 0:
                bp['boxes'][0].set_facecolor('#2ecc71')
            if len(bp['boxes']) > 1:
                bp['boxes'][1].set_facecolor('#e74c3c')
            axes[0, 1].set_ylabel('Total Token Usage')
            axes[0, 1].set_title('Token Usage Distribution by Resolution Status')
            axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Scatter: tokens vs tool usage
        tokens = metrics.tokens_per_traj
        tool_calls = metrics.tool_calls_per_traj
        colors = ['#2ecc71' if t.resolved else '#e74c3c' for t in run.trajectories]
        
        axes[1, 0].scatter(tokens, tool_calls, c=colors, alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Token Usage')
        axes[1, 0].set_ylabel('Tool Calls')
        axes[1, 0].set_title('Token Usage vs Tool Calls')
        axes[1, 0].grid(alpha=0.3)
        
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Resolved'),
            Patch(facecolor='#e74c3c', label='Unresolved')
        ]
        axes[1, 0].legend(handles=legend_elements)
        
        # 4. Token type breakdown (stacked bar for all trajectories)
        all_cache_read = [t.total_cache_read_input_tokens for t in run.trajectories]
        all_cache_creation = [t.total_cache_creation_input_tokens for t in run.trajectories]
        all_input = [t.total_input_tokens for t in run.trajectories]
        all_output = [t.total_output_tokens for t in run.trajectories]
        
        avg_cache_read = np.mean(all_cache_read) if all_cache_read else 0
        avg_cache_creation = np.mean(all_cache_creation) if all_cache_creation else 0
        avg_input = np.mean(all_input) if all_input else 0
        avg_output = np.mean(all_output) if all_output else 0
        
        # Create a single stacked bar
        x_pos = 0
        width = 0.6
        bottom = 0
        
        if avg_cache_read > 0:
            axes[1, 1].bar(x_pos, avg_cache_read, width, bottom=bottom,
                          label='Cache Read Input Tokens', color='#9b59b6', alpha=0.7)
            bottom += avg_cache_read
        
        if avg_cache_creation > 0:
            axes[1, 1].bar(x_pos, avg_cache_creation, width, bottom=bottom,
                         label='Cache Creation Input Tokens', color='#f39c12', alpha=0.7)
            bottom += avg_cache_creation
        
        if avg_input > 0:
            axes[1, 1].bar(x_pos, avg_input, width, bottom=bottom,
                          label='Input Tokens (after cache)', color='#3498db', alpha=0.7)
            bottom += avg_input
        
        if avg_output > 0:
            axes[1, 1].bar(x_pos, avg_output, width, bottom=bottom,
                          label='Output Tokens', color='#e74c3c', alpha=0.7)
            bottom += avg_output
        
        axes[1, 1].set_ylabel('Average Token Usage')
        axes[1, 1].set_title('Average Token Type Breakdown')
        axes[1, 1].set_xticks([])
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add total value label
        total = bottom
        if total > 0:
            axes[1, 1].text(x_pos, total + max([avg_cache_read, avg_cache_creation, avg_input, avg_output]) * 0.02,
                          f'Total: {int(total):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_token_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved token analysis plot to {output_path}")
        return output_path
    
    def plot_error_patterns(
        self, 
        run: Run, 
        top_n: int = 20, 
        mode: str = "raw"
    ) -> Path | None:
        """Plot top tool execution error patterns for a run.
        
        Args:
            run: Run to analyze
            top_n: Number of top error patterns to show
            mode: "categorized" for predefined patterns, "raw" for first 5 words
            
        Returns:
            Path to generated plot or None
        """
        # Collect error patterns from failed tool calls
        error_patterns: dict[str, int] = {}
        tool_error_patterns: dict[str, dict[str, int]] = {}
        
        for traj in run.trajectories:
            for tc in traj.get_tool_calls():
                if not tc.success and tc.result:
                    pattern = extract_error_pattern(tc.result, mode=mode)
                    error_patterns[pattern] = error_patterns.get(pattern, 0) + 1
                    
                    if tc.name not in tool_error_patterns:
                        tool_error_patterns[tc.name] = {}
                    tool_error_patterns[tc.name][pattern] = tool_error_patterns[tc.name].get(pattern, 0) + 1
        
        if not error_patterns:
            logger.warning(f"No error patterns found for {run.name}")
            return None
        
        # Sort and get top N
        sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
        patterns, counts = zip(*sorted_patterns)
        
        fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(patterns) * 0.4)))
        mode_label = "Raw Error Messages" if mode == "raw" else "Categorized Patterns"
        fig.suptitle(f'Tool Execution Errors - {run.name}\n({mode_label})', fontsize=14, fontweight='bold')
        
        # 1. Top error patterns bar chart
        y_pos = np.arange(len(patterns))
        bars = axes[0].barh(y_pos, counts, color='#e74c3c', alpha=0.7)
        axes[0].set_yticks(y_pos)
        # For raw mode, show actual text; for categorized, title-case
        if mode == "raw":
            axes[0].set_yticklabels(patterns, fontsize=8)
        else:
            axes[0].set_yticklabels([p.replace('_', ' ').title() for p in patterns])
        axes[0].invert_yaxis()  # Top pattern at top
        axes[0].set_xlabel('Frequency')
        axes[0].set_title(f'Top {len(patterns)} Error Patterns')
        axes[0].grid(axis='x', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                        str(count), va='center', fontsize=9)
        
        # 2. Error patterns by tool (stacked or grouped)
        # Get top tools by error count
        tool_totals = {tool: sum(patterns.values()) for tool, patterns in tool_error_patterns.items()}
        top_tools = sorted(tool_totals.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if top_tools:
            # Get top 5 patterns for these tools
            top_5_patterns = [p for p, _ in sorted_patterns[:5]]
            
            x = np.arange(len(top_tools))
            width = 0.15
            colors = plt.cm.Set2(np.linspace(0, 1, len(top_5_patterns)))
            
            for i, pattern in enumerate(top_5_patterns):
                tool_counts = [tool_error_patterns.get(tool, {}).get(pattern, 0) for tool, _ in top_tools]
                axes[1].bar(x + i*width, tool_counts, width, label=pattern.replace('_', ' '), color=colors[i], alpha=0.8)
            
            axes[1].set_xticks(x + width * 2)
            axes[1].set_xticklabels([tool for tool, _ in top_tools], rotation=45, ha='right')
            axes[1].set_ylabel('Error Count')
            axes[1].set_title('Top Error Patterns by Tool')
            axes[1].legend(loc='upper right', fontsize=8)
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_error_patterns.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error patterns plot to {output_path}")
        return output_path
    
    def plot_failed_tool_calls(
        self, 
        run: Run, 
        top_n: int = 25,
    ) -> Path | None:
        """Plot the most common failing tool calls with their arguments.
        
        Args:
            run: Run to analyze
            top_n: Number of top failing calls to show
            
        Returns:
            Path to generated plot or None
        """
        # Collect failed tool calls with their signatures
        failed_calls: dict[str, int] = {}
        failed_calls_with_error: dict[str, list[str]] = {}  # signature -> sample errors
        
        for traj in run.trajectories:
            for tc in traj.get_tool_calls():
                if not tc.success:
                    signature = format_tool_call_signature(tc.name, tc.arguments)
                    failed_calls[signature] = failed_calls.get(signature, 0) + 1
                    
                    # Store sample errors for this signature
                    if signature not in failed_calls_with_error:
                        failed_calls_with_error[signature] = []
                    if len(failed_calls_with_error[signature]) < 3 and tc.result:
                        error_preview = extract_raw_error_prefix(tc.result, num_words=8)
                        failed_calls_with_error[signature].append(error_preview)
        
        if not failed_calls:
            logger.warning(f"No failed tool calls found for {run.name}")
            return None
        
        # Sort and get top N
        sorted_calls = sorted(failed_calls.items(), key=lambda x: x[1], reverse=True)[:top_n]
        signatures, counts = zip(*sorted_calls)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, max(10, len(signatures) * 0.5)))
        fig.suptitle(
            f'Most Common Failing Tool Calls - {run.name}\n(with arguments)',
            fontsize=14, fontweight='bold'
        )
        
        # Horizontal bar chart
        y_pos = np.arange(len(signatures))
        bars = ax.barh(y_pos, counts, color='#e74c3c', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(signatures, fontsize=8, fontfamily='monospace')
        ax.invert_yaxis()
        ax.set_xlabel('Failure Count')
        ax.set_title(f'Top {len(signatures)} Failing Tool Calls')
        ax.grid(axis='x', alpha=0.3)
        
        # Add count labels
        max_count = max(counts)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + max_count * 0.01, bar.get_y() + bar.get_height()/2,
                   str(count), va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_failed_tool_calls.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved failed tool calls plot to {output_path}")
        return output_path
    
    def plot_error_analysis(self, run: Run) -> Path | None:
        """Comprehensive error analysis showing main reasons for tool call errors.
        
        Args:
            run: Run to analyze
            
        Returns:
            Path to generated plot or None
        """
        from collections import defaultdict
        
        # Collect error data
        error_categories: dict[str, int] = defaultdict(int)
        tool_error_counts: dict[str, int] = defaultdict(int)
        tool_total_counts: dict[str, int] = defaultdict(int)
        error_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)  # category -> [(tool, error_msg)]
        tool_error_patterns: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for traj in run.trajectories:
            for tc in traj.get_tool_calls():
                tool_total_counts[tc.name] += 1
                
                if not tc.success and tc.result:
                    # Categorize error
                    category = extract_error_pattern(tc.result, mode="categorized")
                    error_categories[category] += 1
                    tool_error_counts[tc.name] += 1
                    
                    # Store sample errors (up to 3 per category)
                    if len(error_samples[category]) < 3:
                        error_preview = (tc.result or "")[:150]
                        error_samples[category].append((tc.name, error_preview))
                    
                    # Track error patterns by tool
                    pattern = extract_error_pattern(tc.result, mode="categorized")
                    tool_error_patterns[tc.name][pattern] += 1
        
        if not error_categories:
            logger.warning(f"No errors found for {run.name}")
            return None
        
        # Calculate error rates
        tool_error_rates = {
            tool: tool_error_counts[tool] / tool_total_counts[tool]
            for tool in tool_total_counts
            if tool_total_counts[tool] > 0
        }
        
        # Sort data
        sorted_categories = sorted(error_categories.items(), key=lambda x: x[1], reverse=True)
        sorted_tools_by_errors = sorted(tool_error_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_tools_by_rate = sorted(tool_error_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Comprehensive Error Analysis - {run.name}', fontsize=16, fontweight='bold')
        
        # 1. Top error categories (pie chart + bar chart)
        top_categories = sorted_categories[:10]
        if top_categories:
            categories, counts = zip(*top_categories)
            
            # Pie chart
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            wedges, texts, autotexts = axes[0, 0].pie(
                counts, labels=categories, autopct='%1.1f%%',
                startangle=90, colors=colors, textprops={'fontsize': 8}
            )
            axes[0, 0].set_title('Error Categories Distribution', fontweight='bold')
            
            # Bar chart with counts
            y_pos = np.arange(len(categories))
            bars = axes[0, 1].barh(y_pos, counts, color=colors, alpha=0.7)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels([c.replace('_', ' ').title() for c in categories], fontsize=9)
            axes[0, 1].invert_yaxis()
            axes[0, 1].set_xlabel('Error Count')
            axes[0, 1].set_title('Top Error Categories (Count)', fontweight='bold')
            axes[0, 1].grid(axis='x', alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, counts):
                axes[0, 1].text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                              str(count), va='center', fontsize=9)
        
        # 2. Error-prone tools (by count and rate)
        top_tools_by_count = sorted_tools_by_errors[:10]
        top_tools_by_rate = [t for t in sorted_tools_by_rate if t[1] > 0][:10]
        
        if top_tools_by_count:
            tools, error_counts = zip(*top_tools_by_count)
            x = np.arange(len(tools))
            width = 0.35
            
            bars = axes[1, 0].bar(x, error_counts, width, color='#e74c3c', alpha=0.7, label='Error Count')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(tools, rotation=45, ha='right', fontsize=9)
            axes[1, 0].set_ylabel('Error Count')
            axes[1, 0].set_title('Most Error-Prone Tools (by Count)', fontweight='bold')
            axes[1, 0].grid(axis='y', alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, error_counts):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                              str(count), ha='center', va='bottom', fontsize=8)
        
        if top_tools_by_rate:
            tools_rate, rates = zip(*top_tools_by_rate)
            x = np.arange(len(tools_rate))
            
            bars = axes[1, 1].barh(x, rates, color='#c0392b', alpha=0.7)
            axes[1, 1].set_yticks(x)
            axes[1, 1].set_yticklabels(tools_rate, fontsize=9)
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_xlabel('Error Rate')
            axes[1, 1].set_title('Tools with Highest Error Rates', fontweight='bold')
            axes[1, 1].set_xlim(0, 1.1)
            axes[1, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            axes[1, 1].grid(axis='x', alpha=0.3)
            
            # Add rate labels
            for bar, rate in zip(bars, rates):
                axes[1, 1].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                              f'{rate:.1%}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / f'{self._safe_name(run.name)}_error_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error analysis plot to {output_path}")
        return output_path
    
    def plot_comparison(self, runs: list[Run]) -> Path | None:
        """Plot comparison across multiple runs."""
        if len(runs) < 2:
            logger.warning("Need at least 2 runs for comparison")
            return None
        
        metrics_dict = self.comparator.compare_runs(runs)
        run_names = list(metrics_dict.keys())
        
        # Adjust figure size based on number of runs to prevent layout issues
        num_runs = len(run_names)
        # Base width and height, increase width for more runs
        fig_width = min(20, 12 + num_runs * 0.5)  # Cap at 20 inches
        fig_height = 14  # Fixed height to prevent excessive expansion
        
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        fig.suptitle('Comparison Across Runs', fontsize=16, fontweight='bold')
        
        # 1. Resolution rate comparison
        resolve_rates = [metrics_dict[name].resolve_rate for name in run_names]
        
        bars = axes[0, 0].bar(range(len(run_names)), resolve_rates, color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Resolution Rate')
        axes[0, 0].set_title('Resolution Rate Comparison')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        axes[0, 0].set_xticks(range(len(run_names)))
        # Truncate long names and adjust rotation based on number of runs
        if num_runs > 10:
            # For many runs, use 90 degree rotation and truncate names
            truncated_names = [name[:20] + '...' if len(name) > 20 else name for name in run_names]
            axes[0, 0].set_xticklabels(truncated_names, rotation=90, ha='right', fontsize=8)
        else:
            axes[0, 0].set_xticklabels(run_names, rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        for i, (bar, rate) in enumerate(zip(bars, resolve_rates)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., rate,
                          f'{rate:.1%}', ha='center', va='bottom')
        
        # 2. Average tool usage comparison
        avg_tool_calls = [metrics_dict[name].avg_tool_calls for name in run_names]
        
        bars = axes[0, 1].bar(range(len(run_names)), avg_tool_calls, color='orange', alpha=0.7)
        axes[0, 1].set_ylabel('Average Tool Calls')
        axes[0, 1].set_title('Average Tool Usage Comparison')
        axes[0, 1].set_xticks(range(len(run_names)))
        if num_runs > 10:
            truncated_names = [name[:20] + '...' if len(name) > 20 else name for name in run_names]
            axes[0, 1].set_xticklabels(truncated_names, rotation=90, ha='right', fontsize=8)
        else:
            axes[0, 1].set_xticklabels(run_names, rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for bar, avg in zip(bars, avg_tool_calls):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., avg,
                          f'{avg:.1f}', ha='center', va='bottom')
        
        # 3. Average tokens comparison (stacked by type)
        # Calculate token breakdowns for each run
        run_cache_read = []
        run_cache_creation = []
        run_input_tokens = []
        run_output_tokens = []
        
        for name in run_names:
            run = next(r for r in runs if r.name == name)
            cache_read_avg = np.mean([t.total_cache_read_input_tokens for t in run.trajectories]) if run.trajectories else 0
            cache_creation_avg = np.mean([t.total_cache_creation_input_tokens for t in run.trajectories]) if run.trajectories else 0
            input_avg = np.mean([t.total_input_tokens for t in run.trajectories]) if run.trajectories else 0
            output_avg = np.mean([t.total_output_tokens for t in run.trajectories]) if run.trajectories else 0
            
            run_cache_read.append(cache_read_avg)
            run_cache_creation.append(cache_creation_avg)
            run_input_tokens.append(input_avg)
            run_output_tokens.append(output_avg)
        
        x = np.arange(len(run_names))
        width = 0.6
        
        # Stack input tokens: cache_read (bottom), cache_creation (middle), regular input (top)
        bottom = np.zeros(len(run_names))
        if any(run_cache_read):
            axes[1, 0].bar(x, run_cache_read, width, bottom=bottom,
                          label='Cache Read Input Tokens', color='#9b59b6', alpha=0.7)
            bottom += np.array(run_cache_read)
        
        if any(run_cache_creation):
            axes[1, 0].bar(x, run_cache_creation, width, bottom=bottom,
                          label='Cache Creation Input Tokens', color='#f39c12', alpha=0.7)
            bottom += np.array(run_cache_creation)
        
        axes[1, 0].bar(x, run_input_tokens, width, bottom=bottom,
                      label='Input Tokens (after cache)', color='#3498db', alpha=0.7)
        bottom += np.array(run_input_tokens)
        
        # Output tokens on top of all input
        axes[1, 0].bar(x, run_output_tokens, width, bottom=bottom,
                      label='Output Tokens', color='#e74c3c', alpha=0.7)
        
        axes[1, 0].set_ylabel('Average Tokens')
        axes[1, 0].set_title('Average Token Usage Comparison (by Type)')
        axes[1, 0].set_xticks(x)
        if num_runs > 10:
            truncated_names = [name[:20] + '...' if len(name) > 20 else name for name in run_names]
            axes[1, 0].set_xticklabels(truncated_names, rotation=90, ha='right', fontsize=8)
        else:
            axes[1, 0].set_xticklabels(run_names, rotation=45, ha='right')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add total value labels on top of stacked bars
        avg_tokens = [metrics_dict[name].avg_tokens for name in run_names]
        for i, (x_pos, total) in enumerate(zip(x, avg_tokens)):
            axes[1, 0].text(x_pos, total + max(avg_tokens) * 0.02,
                          f'{total:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Scaffold comparison (group by scaffold)
        scaffold_metrics: dict[str, list[float]] = {}
        for name in run_names:
            scaffold = metrics_dict[name].scaffold
            if scaffold not in scaffold_metrics:
                scaffold_metrics[scaffold] = []
            scaffold_metrics[scaffold].append(metrics_dict[name].resolve_rate)
        
        scaffolds = list(scaffold_metrics.keys())
        avg_by_scaffold = [np.mean(scaffold_metrics[s]) for s in scaffolds]
        
        bars = axes[1, 1].bar(scaffolds, avg_by_scaffold, color='green', alpha=0.7)
        axes[1, 1].set_ylabel('Average Resolution Rate')
        axes[1, 1].set_title('Resolution Rate by Scaffold')
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        for bar, avg in zip(bars, avg_by_scaffold):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., avg,
                          f'{avg:.1%}', ha='center', va='bottom')
        
        # Use constrained_layout instead of tight_layout for better handling of many runs
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        output_path = self.output_dir / 'comparison_across_runs.png'
        # Use bbox_inches='tight' but with a reasonable DPI to prevent excessive size
        # If still too large, fall back to regular bbox
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        except (ValueError, OverflowError):
            # Fallback: save without tight bbox if size is still too large
            logger.warning("Figure too large for tight bbox, saving with regular bbox")
            plt.savefig(output_path, dpi=150, facecolor='white')
        plt.close()
        logger.info(f"Saved comparison plot to {output_path}")
        return output_path
    
    def plot_transfer_analysis(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> Path | None:
        """Plot transfer learning analysis.
        
        Args:
            runs: List of runs to analyze
            model_to_trained_scaffold: Mapping of models to their training scaffold
        """
        if model_to_trained_scaffold is None:
            # Default known mappings
            model_to_trained_scaffold = {
                "agentica-org/DeepSWE-Preview": "r2e-gym",
                "mistralai/devstral-2512:free": "mistral-vibe-cli",
                "anthropic/claude-3-5-haiku-20241022": "claude-code",
            }
        
        analyses = self.comparator.find_transfer_pairs(runs, model_to_trained_scaffold)
        
        if not analyses:
            logger.warning("No transfer pairs found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Transfer Learning Analysis', fontsize=16, fontweight='bold')
        
        # 1. Transfer delta by model/scaffold
        labels = [f"{a.base_model.split('/')[-1]}\n{a.source_scaffold}→{a.target_scaffold}" 
                  for a in analyses]
        deltas = [a.transfer_delta for a in analyses]
        colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in deltas]
        
        bars = axes[0].barh(range(len(labels)), deltas, color=colors, alpha=0.7)
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels)
        axes[0].set_xlabel('Resolution Rate Delta')
        axes[0].set_title('Transfer Learning Impact')
        axes[0].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:+.1%}'))
        axes[0].grid(axis='x', alpha=0.3)
        
        # 2. Source vs target performance
        source_rates = [a.source_resolve_rate for a in analyses]
        target_rates = [a.target_resolve_rate for a in analyses]
        
        x = np.arange(len(analyses))
        width = 0.35
        
        bars1 = axes[1].bar(x - width/2, source_rates, width, label='Source Scaffold', color='#3498db', alpha=0.7)
        bars2 = axes[1].bar(x + width/2, target_rates, width, label='Target Scaffold', color='#9b59b6', alpha=0.7)
        
        axes[1].set_ylabel('Resolution Rate')
        axes[1].set_title('Source vs Target Performance')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([a.base_model.split('/')[-1] for a in analyses], rotation=45, ha='right')
        axes[1].set_ylim(0, 1.1)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'transfer_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved transfer analysis plot to {output_path}")
        return output_path
    
    def _safe_name(self, name: str) -> str:
        """Convert run name to safe filename."""
        return name.replace('/', '_').replace(' ', '_').replace(':', '_')
       
    def _generate_transfer_analyses(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> list[DeepTransferAnalysis]:
        """Compute deep transfer analyses for all source→target pairs."""
        if model_to_trained_scaffold is None:
            model_to_trained_scaffold = {
                "agentica-org/DeepSWE-Preview": "r2e-gym",
                "mistralai/devstral-2512:free": "mistral-vibe-cli",
                "anthropic/claude-3-5-haiku-20241022": "claude-code",
            }
        
        from collections import defaultdict
        
        runs_by_model: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            runs_by_model[run.base_model].append(run)
        
        analyzer = TransferAnalyzer()
        analyses: list[DeepTransferAnalysis] = []
        
        for model, model_runs in runs_by_model.items():
            trained_scaffold = model_to_trained_scaffold.get(model)
            if not trained_scaffold:
                continue
            
            source_runs = [r for r in model_runs if r.scaffold == trained_scaffold]
            if not source_runs:
                continue
            
            source_run = source_runs[0]
            
            for target_run in model_runs:
                if target_run.scaffold == trained_scaffold:
                    continue
                
                logger.info(
                    f"Generating deep transfer analysis: {source_run.name} -> {target_run.name}"
                )
                analyses.append(analyzer.analyze_transfer(source_run, target_run))
        
        return analyses
    
    def plot_transfer_mcnemar_all(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> list[Path]:
        """Generate McNemar transfer plots for all transfer pairs."""
        generated: list[Path] = []
        for analysis in self._generate_transfer_analyses(runs, model_to_trained_scaffold):
            path = self._plot_mcnemar(analysis)
            if path:
                generated.append(path)
        return generated
    
    def plot_transfer_error_modes_all(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> list[Path]:
        """Generate transfer error-mode plots for all transfer pairs."""
        generated: list[Path] = []
        for analysis in self._generate_transfer_analyses(runs, model_to_trained_scaffold):
            path = self._plot_error_modes(analysis)
            if path:
                generated.append(path)
        return generated
    
    def plot_transfer_vocabulary_all(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> list[Path]:
        """Generate transfer vocabulary plots for all transfer pairs."""
        generated: list[Path] = []
        for analysis in self._generate_transfer_analyses(runs, model_to_trained_scaffold):
            path = self._plot_vocabulary(analysis)
            if path:
                generated.append(path)
        return generated
    
    def plot_transfer_error_patterns_all(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str] | None = None,
    ) -> list[Path]:
        """Generate transfer error-pattern plots for all transfer pairs."""
        generated: list[Path] = []
        for analysis in self._generate_transfer_analyses(runs, model_to_trained_scaffold):
            path = self._plot_error_patterns_comparison(analysis)
            if path:
                generated.append(path)
        return generated
    
    def _plot_mcnemar(self, analysis: DeepTransferAnalysis) -> Path | None:
        """Plot McNemar's test results with contingency table."""
        mcnemar = analysis.mcnemar
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'Instance-Level Transfer Analysis: {analysis.source_scaffold} → {analysis.target_scaffold}\n'
            f'Model: {analysis.base_model}',
            fontsize=14, fontweight='bold'
        )
        
        # 1. Contingency table heatmap
        contingency = np.array([
            [mcnemar.both_success, mcnemar.target_only],
            [mcnemar.source_only, mcnemar.both_failure]
        ])
        
        im = axes[0].imshow(contingency, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                val = contingency[i, j]
                pct = val / mcnemar.total_instances * 100 if mcnemar.total_instances > 0 else 0
                axes[0].text(j, i, f'{val}\n({pct:.1f}%)', ha='center', va='center', 
                           fontsize=12, fontweight='bold')
        
        axes[0].set_xticks([0, 1])
        axes[0].set_yticks([0, 1])
        axes[0].set_xticklabels(['Target ✓', 'Target ✗'])
        axes[0].set_yticklabels(['Source ✓', 'Source ✗'])
        axes[0].set_xlabel('Target Scaffold')
        axes[0].set_ylabel('Source Scaffold')
        axes[0].set_title('Contingency Table')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[0], label='Count')
        
        # 2. Transfer outcome breakdown (pie chart)
        labels = ['Both Success', 'Transfer Loss\n(Source only)', 
                  'Transfer Gain\n(Target only)', 'Both Failure']
        sizes = [mcnemar.both_success, mcnemar.source_only, 
                mcnemar.target_only, mcnemar.both_failure]
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
        explode = (0, 0.05, 0.05, 0)  # Highlight discordant pairs
        
        # Filter out zero values for cleaner pie
        non_zero = [(l, s, c, e) for l, s, c, e in zip(labels, sizes, colors, explode) if s > 0]
        if non_zero:
            labels, sizes, colors, explode = zip(*non_zero)
            axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Transfer Outcome Distribution')
        
        # 3. Statistical summary
        axes[2].axis('off')
        
        # Build summary text
        sig_text = "Yes (p < 0.05)" if mcnemar.is_significant else "No (p ≥ 0.05)"
        direction = mcnemar.transfer_direction.capitalize()
        
        summary_text = f"""
        McNemar's Test Results
        ══════════════════════════════
        
        Total Paired Instances: {mcnemar.total_instances}
        
        Source Resolution Rate: {mcnemar.source_resolve_rate:.1%}
        Target Resolution Rate: {mcnemar.target_resolve_rate:.1%}
        
        Discordant Pairs:
          • Transfer Loss (source→fail): {mcnemar.source_only}
          • Transfer Gain (fail→target): {mcnemar.target_only}
        
        Chi-squared Statistic: {mcnemar.chi_squared:.3f}
        P-value: {mcnemar.p_value:.4f}
        
        Statistically Significant: {sig_text}
        Transfer Direction: {direction}
        """
        
        axes[2].text(0.1, 0.5, summary_text, transform=axes[2].transAxes,
                    fontsize=11, verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title('Statistical Summary')
        
        plt.tight_layout()
        output_path = self.output_dir / f'transfer_mcnemar_{self._safe_name(analysis.source_scaffold)}_{self._safe_name(analysis.target_scaffold)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved McNemar analysis plot to {output_path}")
        return output_path
    
    def _plot_error_modes(self, analysis: DeepTransferAnalysis) -> Path | None:
        """Plot error mode comparison between scaffolds."""
        if not analysis.source_errors or not analysis.target_errors:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Error Mode Analysis: {analysis.source_scaffold} vs {analysis.target_scaffold}',
            fontsize=14, fontweight='bold'
        )
        
        source_errors = analysis.source_errors
        target_errors = analysis.target_errors
        
        # 1. Error category comparison
        all_categories = set(source_errors.error_counts.keys()) | set(target_errors.error_counts.keys())
        categories = sorted(all_categories)
        
        if categories:
            source_counts = [source_errors.error_counts.get(c, 0) for c in categories]
            target_counts = [target_errors.error_counts.get(c, 0) for c in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            bars1 = axes[0, 0].bar(x - width/2, source_counts, width, 
                                   label=f'Source ({analysis.source_scaffold})', color='#3498db', alpha=0.7)
            bars2 = axes[0, 0].bar(x + width/2, target_counts, width,
                                   label=f'Target ({analysis.target_scaffold})', color='#e74c3c', alpha=0.7)
            
            # Clean up category names for display
            display_names = [c.replace('_', ' ').title() for c in categories]
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(display_names, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Error Categories Comparison')
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Error-prone tools comparison
        source_tools = dict(source_errors.error_prone_tools[:8])
        target_tools = dict(target_errors.error_prone_tools[:8])
        all_tools = sorted(set(source_tools.keys()) | set(target_tools.keys()),
                          key=lambda t: source_tools.get(t, 0) + target_tools.get(t, 0),
                          reverse=True)[:10]
        
        if all_tools:
            source_tool_counts = [source_tools.get(t, 0) for t in all_tools]
            target_tool_counts = [target_tools.get(t, 0) for t in all_tools]
            
            x = np.arange(len(all_tools))
            width = 0.35
            
            axes[0, 1].barh(x - width/2, source_tool_counts, width,
                           label=f'Source', color='#3498db', alpha=0.7)
            axes[0, 1].barh(x + width/2, target_tool_counts, width,
                           label=f'Target', color='#e74c3c', alpha=0.7)
            
            axes[0, 1].set_yticks(x)
            axes[0, 1].set_yticklabels(all_tools)
            axes[0, 1].set_xlabel('Error Count')
            axes[0, 1].set_title('Error-Prone Tools')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='x', alpha=0.3)
        
        # 3. Tool error rates comparison
        source_rates = source_errors.tool_error_rates
        target_rates = target_errors.tool_error_rates
        all_rate_tools = sorted(set(source_rates.keys()) | set(target_rates.keys()),
                                key=lambda t: max(source_rates.get(t, 0), target_rates.get(t, 0)),
                                reverse=True)[:10]
        
        if all_rate_tools:
            source_rate_vals = [source_rates.get(t, 0) for t in all_rate_tools]
            target_rate_vals = [target_rates.get(t, 0) for t in all_rate_tools]
            
            x = np.arange(len(all_rate_tools))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, source_rate_vals, width,
                          label=f'Source', color='#3498db', alpha=0.7)
            axes[1, 0].bar(x + width/2, target_rate_vals, width,
                          label=f'Target', color='#e74c3c', alpha=0.7)
            
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(all_rate_tools, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Error Rate')
            axes[1, 0].set_title('Tool Error Rates')
            axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Summary statistics
        axes[1, 1].axis('off')
        
        summary_text = f"""
        Error Summary
        ═══════════════════════════════
        
        Source ({analysis.source_scaffold}):
          Total Errors: {source_errors.total_errors}
          Most Common: {source_errors.error_prone_tools[0][0] if source_errors.error_prone_tools else 'N/A'}
        
        Target ({analysis.target_scaffold}):
          Total Errors: {target_errors.total_errors}
          Most Common: {target_errors.error_prone_tools[0][0] if target_errors.error_prone_tools else 'N/A'}
        
        Error Increase on Target: {target_errors.total_errors - source_errors.total_errors:+d}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=11, verticalalignment='center', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / f'transfer_error_modes_{self._safe_name(analysis.source_scaffold)}_{self._safe_name(analysis.target_scaffold)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error mode analysis plot to {output_path}")
        return output_path
    
    def _plot_vocabulary(self, analysis: DeepTransferAnalysis) -> Path | None:
        """Plot tool vocabulary alignment analysis."""
        if not analysis.vocabulary:
            return None
        
        vocab = analysis.vocabulary
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Tool Vocabulary Analysis: {vocab.source_scaffold} vs {vocab.target_scaffold}',
            fontsize=14, fontweight='bold'
        )
        
        # 1. Vocabulary overlap (horizontal bar showing sets)
        ax = axes[0, 0]
        
        # Create a simple Venn-like visualization
        total_source = len(vocab.source_tools)
        total_target = len(vocab.target_tools)
        shared = len(vocab.shared_tools)
        source_only = len(vocab.source_only_tools)
        target_only = len(vocab.target_only_tools)
        
        categories = ['Source Only', 'Shared', 'Target Only']
        values = [source_only, shared, target_only]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax.barh(categories, values, color=colors, alpha=0.7)
        ax.set_xlabel('Number of Tools')
        ax.set_title(f'Tool Vocabulary Overlap (Jaccard: {vocab.jaccard_index:.2f})')
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   str(val), va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        
        # 2. Tool usage distribution comparison
        ax = axes[0, 1]
        
        # Get top tools from each
        source_sorted = sorted(vocab.source_tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        target_sorted = sorted(vocab.target_tool_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        all_top_tools = []
        seen = set()
        for tool, _ in source_sorted + target_sorted:
            if tool not in seen:
                all_top_tools.append(tool)
                seen.add(tool)
        all_top_tools = all_top_tools[:12]
        
        if all_top_tools:
            source_vals = [vocab.source_tool_counts.get(t, 0) for t in all_top_tools]
            target_vals = [vocab.target_tool_counts.get(t, 0) for t in all_top_tools]
            
            x = np.arange(len(all_top_tools))
            width = 0.35
            
            ax.bar(x - width/2, source_vals, width, label='Source', color='#3498db', alpha=0.7)
            ax.bar(x + width/2, target_vals, width, label='Target', color='#e74c3c', alpha=0.7)
            
            ax.set_xticks(x)
            ax.set_xticklabels(all_top_tools, rotation=45, ha='right')
            ax.set_ylabel('Call Count')
            ax.set_title('Tool Usage Comparison')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # 3. Hallucinated tools
        ax = axes[1, 0]
        
        if vocab.hallucinated_tools:
            tools = list(vocab.hallucinated_tools.keys())
            counts = list(vocab.hallucinated_tools.values())
            
            bars = ax.barh(range(len(tools)), counts, color='#e74c3c', alpha=0.7)
            ax.set_yticks(range(len(tools)))
            ax.set_yticklabels(tools)
            ax.set_xlabel('Call Count')
            ax.set_title(f'Hallucinated Tools on Target\n(Rate: {vocab.hallucination_rate:.1%})')
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hallucinated tools detected',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Hallucinated Tools on Target')
            ax.axis('off')
        
        # 4. Tool mapping suggestions
        ax = axes[1, 1]
        ax.axis('off')
        
        # Build tool list text
        source_only_text = ', '.join(sorted(vocab.source_only_tools)[:10]) or 'None'
        target_only_text = ', '.join(sorted(vocab.target_only_tools)[:10]) or 'None'
        shared_text = ', '.join(sorted(vocab.shared_tools)[:10]) or 'None'
        
        mapping_text = '\n'.join([f"  {s} → {t}" for s, t in vocab.tool_mappings.items()]) or '  (none defined)'
        
        summary_text = f"""
        Tool Vocabulary Summary
        ════════════════════════════════════
        
        Source Tools ({len(vocab.source_tools)}):
          {source_only_text}
        
        Target Tools ({len(vocab.target_tools)}):
          {target_only_text}
        
        Shared Tools ({len(vocab.shared_tools)}):
          {shared_text}
        
        Known Tool Mappings:
        {mapping_text}
        
        Jaccard Similarity: {vocab.jaccard_index:.2f}
        Hallucination Rate: {vocab.hallucination_rate:.1%}
        """
        
        ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / f'transfer_vocabulary_{self._safe_name(vocab.source_scaffold)}_{self._safe_name(vocab.target_scaffold)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved vocabulary analysis plot to {output_path}")
        return output_path
    
    def _plot_error_patterns_comparison(
        self, 
        analysis: DeepTransferAnalysis,
        top_n: int = 20
    ) -> Path | None:
        """Plot error pattern comparison between source and target scaffolds."""
        if not analysis.source_errors or not analysis.target_errors:
            return None
        
        source_patterns = analysis.source_errors.error_pattern_counts
        target_patterns = analysis.target_errors.error_pattern_counts
        
        if not source_patterns and not target_patterns:
            logger.warning("No error patterns to compare")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(20, max(10, len(source_patterns) * 0.5)))
        fig.suptitle(
            f'Error Pattern Comparison: {analysis.source_scaffold} vs {analysis.target_scaffold}\n'
            f'Model: {analysis.base_model} (Raw Error Messages)',
            fontsize=14, fontweight='bold'
        )
        
        # Combine and sort patterns by total frequency
        all_patterns = set(source_patterns.keys()) | set(target_patterns.keys())
        pattern_totals = {
            p: source_patterns.get(p, 0) + target_patterns.get(p, 0) 
            for p in all_patterns
        }
        top_patterns = sorted(pattern_totals.items(), key=lambda x: x[1], reverse=True)[:top_n]
        patterns = [p for p, _ in top_patterns]
        
        # 1. Side-by-side comparison of top patterns
        source_counts = [source_patterns.get(p, 0) for p in patterns]
        target_counts = [target_patterns.get(p, 0) for p in patterns]
        
        y_pos = np.arange(len(patterns))
        height = 0.35
        
        bars1 = axes[0].barh(y_pos - height/2, source_counts, height, 
                            label=f'Source ({analysis.source_scaffold})', color='#3498db', alpha=0.7)
        bars2 = axes[0].barh(y_pos + height/2, target_counts, height,
                            label=f'Target ({analysis.target_scaffold})', color='#e74c3c', alpha=0.7)
        
        axes[0].set_yticks(y_pos)
        # Show raw error messages (not title-cased)
        axes[0].set_yticklabels(patterns, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Frequency')
        axes[0].set_title(f'Top {len(patterns)} Error Messages Comparison')
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='x', alpha=0.3)
        
        # 2. Delta plot (target - source) to show what increased/decreased
        deltas = [target_counts[i] - source_counts[i] for i in range(len(patterns))]
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in deltas]
        
        bars = axes[1].barh(y_pos, deltas, color=colors, alpha=0.7)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(patterns, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Delta (Target - Source)')
        axes[1].set_title('Error Pattern Change\n(Red = More on Target, Blue = More on Source)')
        axes[1].axvline(0, color='black', linestyle='-', linewidth=0.5)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add annotations for significant changes
        max_delta = max(abs(d) for d in deltas) if deltas else 1
        for bar, delta in zip(bars, deltas):
            if abs(delta) > max_delta * 0.1:  # Only annotate significant changes
                sign = '+' if delta > 0 else ''
                axes[1].text(
                    delta + (max_delta * 0.02 if delta >= 0 else -max_delta * 0.02),
                    bar.get_y() + bar.get_height()/2,
                    f'{sign}{delta}',
                    va='center',
                    ha='left' if delta >= 0 else 'right',
                    fontsize=8
                )
        
        plt.tight_layout()
        output_path = self.output_dir / f'transfer_error_patterns_{self._safe_name(analysis.source_scaffold)}_{self._safe_name(analysis.target_scaffold)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error patterns comparison plot to {output_path}")
        return output_path
    
    def plot_variance_vs_difficulty_all(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[Path]:
        """Generate variance vs difficulty plots for all scaffold-model pairs.
        
        Args:
            runs: List of runs to analyze
            instances: Optional dictionary mapping instance_id to Instance objects
            
        Returns:
            List of generated plot file paths
        """
        analyzer = VarianceAnalyzer()
        analyses = analyzer.analyze_runs(runs, instances)
        
        generated: list[Path] = []
        for analysis in analyses:
            path = self.plot_variance_vs_difficulty(analysis)
            if path:
                generated.append(path)
        
        return generated
    
    def plot_variance_vs_difficulty(
        self,
        analysis: VarianceAnalysis,
    ) -> Path | None:
        """Plot variance vs difficulty metrics for a scaffold-model pair.
        
        Args:
            analysis: VarianceAnalysis object
            
        Returns:
            Path to generated plot or None
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("pandas is required for variance plotting")
            return None
        
        df = analysis.to_dataframe()
        
        if not isinstance(df, pd.DataFrame):
            logger.error("DataFrame conversion failed")
            return None
        
        # Filter out instances without difficulty metrics
        df_with_difficulty = df[
            df['num_files_changed'].notna() & df['num_lines_changed'].notna()
        ]
        
        if len(df_with_difficulty) == 0:
            logger.warning(
                f"No instances with difficulty metrics for {analysis.scaffold}/{analysis.base_model}"
            )
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Variance vs Difficulty Analysis: {analysis.scaffold} / {analysis.base_model}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Files changed vs resolve_rate (scatter)
        ax = axes[0, 0]
        scatter = ax.scatter(
            df_with_difficulty['num_files_changed'],
            df_with_difficulty['resolve_rate'],
            c=df_with_difficulty['resolve_variance'],
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
        ax.set_xlabel('Number of Files Changed')
        ax.set_ylabel('Resolution Rate')
        ax.set_title(f'Files Changed vs Resolution Rate\n(Correlation: {analysis.correlation_files_resolve_rate:.3f})')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Variance')
        
        # Add correlation annotation
        if analysis.correlation_files_resolve_rate is not None:
            corr = analysis.correlation_files_resolve_rate
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 2. Lines changed vs resolve_rate (scatter)
        ax = axes[0, 1]
        scatter = ax.scatter(
            df_with_difficulty['num_lines_changed'],
            df_with_difficulty['resolve_rate'],
            c=df_with_difficulty['resolve_variance'],
            cmap='viridis',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
        ax.set_xlabel('Number of Lines Changed')
        ax.set_ylabel('Resolution Rate')
        ax.set_title(f'Lines Changed vs Resolution Rate\n(Correlation: {analysis.correlation_lines_resolve_rate:.3f})')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Variance')
        
        if analysis.correlation_lines_resolve_rate is not None:
            corr = analysis.correlation_lines_resolve_rate
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Files changed vs variance (scatter)
        ax = axes[1, 0]
        scatter = ax.scatter(
            df_with_difficulty['num_files_changed'],
            df_with_difficulty['resolve_variance'],
            c=df_with_difficulty['resolve_rate'],
            cmap='RdYlGn',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
        ax.set_xlabel('Number of Files Changed')
        ax.set_ylabel('Resolution Variance')
        ax.set_title(f'Files Changed vs Variance\n(Correlation: {analysis.correlation_files_variance:.3f})')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Resolution Rate')
        
        if analysis.correlation_files_variance is not None:
            corr = analysis.correlation_files_variance
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Lines changed vs variance (scatter)
        ax = axes[1, 1]
        scatter = ax.scatter(
            df_with_difficulty['num_lines_changed'],
            df_with_difficulty['resolve_variance'],
            c=df_with_difficulty['resolve_rate'],
            cmap='RdYlGn',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidths=0.5,
        )
        ax.set_xlabel('Number of Lines Changed')
        ax.set_ylabel('Resolution Variance')
        ax.set_title(f'Lines Changed vs Variance\n(Correlation: {analysis.correlation_lines_variance:.3f})')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Resolution Rate')
        
        if analysis.correlation_lines_variance is not None:
            corr = analysis.correlation_lines_variance
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / f'variance_vs_difficulty_{self._safe_name(analysis.scaffold)}_{self._safe_name(analysis.base_model)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved variance vs difficulty plot to {output_path}")
        return output_path
    
    def plot_deep_variance_all(
        self,
        runs: list[Run],
    ) -> list[Path]:
        """Generate deep variance plots for all scaffold-model pairs.
        
        Args:
            runs: List of runs to analyze
            
        Returns:
            List of generated plot file paths
        """
        analyzer = VarianceAnalyzer()
        analyses = analyzer.analyze_deep_variance(runs)
        
        generated: list[Path] = []
        for analysis in analyses:
            path = self.plot_deep_variance(analysis)
            if path:
                generated.append(path)
        
        return generated
    
    def plot_deep_variance(
        self,
        analysis: DeepVarianceAnalysis,
    ) -> Path | None:
        """Plot comprehensive variance analysis showing mean ± stddev for all metrics.
        
        Args:
            analysis: DeepVarianceAnalysis object
            
        Returns:
            Path to generated plot or None
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(
            f'Deep Variance Analysis: {analysis.scaffold} / {analysis.base_model}\n'
            f'({analysis.num_runs} runs)',
            fontsize=16, fontweight='bold'
        )
        
        # Helper function to create bar plot with error bars
        def plot_metric(ax, stat: RunMetricsStatistics, title: str, ylabel: str, 
                       color: str = 'steelblue', format_func=None):
            """Plot a metric with mean ± stddev."""
            if not stat.values:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title)
                return
            
            mean = stat.mean
            stddev = stat.stddev
            
            bars = ax.barh([0], [mean], color=color, alpha=0.7, xerr=stddev, capsize=5)
            ax.set_xlabel(ylabel)
            ax.set_title(title)
            ax.set_yticks([])
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            if format_func:
                label = f'{format_func(mean)} ± {format_func(stddev)}'
            else:
                label = f'{mean:.2f} ± {stddev:.2f}'
            ax.text(mean, 0, label, ha='left' if mean >= 0 else 'right', 
                   va='center', fontweight='bold', fontsize=10)
        
        # Row 1: Resolution and Patch Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        plot_metric(ax1, analysis.resolve_rate, 'Resolution Rate', 'Rate', 
                   color='#2ecc71', format_func=lambda x: f'{x:.1%}')
        
        ax2 = fig.add_subplot(gs[0, 1])
        plot_metric(ax2, analysis.patch_rate, 'Patch Generation Rate', 'Rate',
                   color='#3498db', format_func=lambda x: f'{x:.1%}')
        
        ax3 = fig.add_subplot(gs[0, 2])
        plot_metric(ax3, analysis.total_instances, 'Total Instances', 'Count',
                   color='#9b59b6', format_func=lambda x: f'{x:.0f}')
        
        # Row 2: Tool Usage Metrics
        ax4 = fig.add_subplot(gs[1, 0])
        plot_metric(ax4, analysis.avg_tool_calls, 'Avg Tool Calls per Trajectory', 'Count',
                   color='#f39c12')
        
        ax5 = fig.add_subplot(gs[1, 1])
        plot_metric(ax5, analysis.median_tool_calls, 'Median Tool Calls per Trajectory', 'Count',
                   color='#e67e22')
        
        ax6 = fig.add_subplot(gs[1, 2])
        plot_metric(ax6, analysis.total_tool_calls, 'Total Tool Calls', 'Count',
                   color='#d35400', format_func=lambda x: f'{x:.0f}')
        
        # Row 3: Token and Step Metrics
        ax7 = fig.add_subplot(gs[2, 0])
        plot_metric(ax7, analysis.avg_tokens, 'Avg Tokens per Trajectory', 'Tokens',
                   color='#1abc9c', format_func=lambda x: f'{x:.0f}')
        
        ax8 = fig.add_subplot(gs[2, 1])
        if analysis.avg_input_tokens.values:
            plot_metric(ax8, analysis.avg_input_tokens, 'Avg Input Tokens', 'Tokens',
                       color='#16a085', format_func=lambda x: f'{x:.0f}')
        else:
            ax8.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax8.transAxes)
            ax8.set_title('Avg Input Tokens')
        
        ax9 = fig.add_subplot(gs[2, 2])
        plot_metric(ax9, analysis.avg_steps, 'Avg Steps per Trajectory', 'Steps',
                   color='#27ae60')
        
        # Create a second figure for resolved vs unresolved comparison
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
        fig2.suptitle(
            f'Resolved vs Unresolved Comparison: {analysis.scaffold} / {analysis.base_model}',
            fontsize=16, fontweight='bold'
        )
        
        # Resolved vs unresolved tool calls
        ax = axes2[0, 0]
        if analysis.resolved_avg_tool_calls.values and analysis.unresolved_avg_tool_calls.values:
            categories = ['Resolved', 'Unresolved']
            means = [analysis.resolved_avg_tool_calls.mean, analysis.unresolved_avg_tool_calls.mean]
            stddevs = [analysis.resolved_avg_tool_calls.stddev, analysis.unresolved_avg_tool_calls.stddev]
            ax.bar(categories, means, yerr=stddevs, capsize=5, 
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
            ax.set_ylabel('Avg Tool Calls')
            ax.set_title('Tool Calls: Resolved vs Unresolved')
            ax.grid(axis='y', alpha=0.3)
            for i, (mean, stddev) in enumerate(zip(means, stddevs)):
                ax.text(i, mean + stddev,
                       f'{mean:.1f} ± {stddev:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Resolved vs unresolved tokens
        ax = axes2[0, 1]
        if analysis.resolved_avg_tokens.values and analysis.unresolved_avg_tokens.values:
            categories = ['Resolved', 'Unresolved']
            means = [analysis.resolved_avg_tokens.mean, analysis.unresolved_avg_tokens.mean]
            stddevs = [analysis.resolved_avg_tokens.stddev, analysis.unresolved_avg_tokens.stddev]
            ax.bar(categories, means, yerr=stddevs, capsize=5,
                   color=['#2ecc71', '#e74c3c'], alpha=0.7)
            ax.set_ylabel('Avg Tokens')
            ax.set_title('Tokens: Resolved vs Unresolved')
            ax.grid(axis='y', alpha=0.3)
            for i, (mean, stddev) in enumerate(zip(means, stddevs)):
                ax.text(i, mean + stddev,
                       f'{mean:.0f} ± {stddev:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Tool success rates (top tools)
        ax = axes2[1, 0]
        if analysis.tool_success_rates:
            sorted_tools = sorted(
                analysis.tool_success_rates.items(),
                key=lambda x: x[1].mean,
                reverse=True
            )[:10]
            
            tools = [t[0] for t in sorted_tools]
            means = [t[1].mean for t in sorted_tools]
            stddevs = [t[1].stddev for t in sorted_tools]
            
            y_pos = np.arange(len(tools))
            bars = ax.barh(y_pos, means, xerr=stddevs, capsize=3,
                           color='#3498db', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tools)
            ax.set_xlabel('Success Rate')
            ax.set_title('Tool Success Rates (Top 10)')
            ax.set_xlim(0, 1.1)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, mean, stddev) in enumerate(zip(bars, means, stddevs)):
                ax.text(mean + stddev, i, f'{mean:.1%} ± {stddev:.1%}',
                       va='center', fontsize=8)
        
        # Tool usage counts (top tools)
        ax = axes2[1, 1]
        if analysis.tool_counts_mean:
            sorted_tools = sorted(
                analysis.tool_counts_mean.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            tools = [t[0] for t in sorted_tools]
            means = [t[1] for t in sorted_tools]
            stddevs = [analysis.tool_counts_stddev.get(t[0], 0.0) for t in sorted_tools]
            
            y_pos = np.arange(len(tools))
            bars = ax.barh(y_pos, means, xerr=stddevs, capsize=3,
                           color='#9b59b6', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(tools)
            ax.set_xlabel('Usage Count')
            ax.set_title('Tool Usage Counts (Top 10)')
            ax.grid(axis='x', alpha=0.3)
            
            for i, (bar, mean, stddev) in enumerate(zip(bars, means, stddevs)):
                ax.text(mean + stddev, i, f'{mean:.0f} ± {stddev:.0f}',
                       va='center', fontsize=8)
        
        plt.tight_layout()
        output_path = self.output_dir / f'deep_variance_{self._safe_name(analysis.scaffold)}_{self._safe_name(analysis.base_model)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save second figure
        plt.tight_layout()
        output_path2 = self.output_dir / f'deep_variance_comparison_{self._safe_name(analysis.scaffold)}_{self._safe_name(analysis.base_model)}.png'
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved deep variance plots to {output_path} and {output_path2}")
        return output_path
    
    # ==================== Trajectory Dynamics Plots ====================
    
    def plot_trajectory_dynamics_all(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[Path]:
        """Generate trajectory dynamics plots for all runs.
        
        Args:
            runs: List of runs to analyze
            instances: Optional instances dict for ground truth
            
        Returns:
            List of generated plot file paths
        """
        analyzer = TrajectoryDynamicsAnalyzer()
        analyses = analyzer.analyze_runs(runs, instances)
        
        generated: list[Path] = []
        for analysis in analyses:
            path = self.plot_trajectory_dynamics(analysis)
            if path:
                generated.append(path)
        
        return generated
    
    def plot_trajectory_dynamics(
        self,
        analysis: TrajectoryDynamicsAnalysis,
    ) -> Path | None:
        """Plot comprehensive trajectory dynamics analysis.
        
        Args:
            analysis: TrajectoryDynamicsAnalysis object
            
        Returns:
            Path to generated plot or None
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f'Trajectory Dynamics: {analysis.run_name}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Localization metrics
        ax = axes[0, 0]
        metrics = ['Touch\nPrecision', 'Touch\nRecall', 'Edit\nPrecision', 'Edit\nRecall']
        values = [
            analysis.mean_touch_precision,
            analysis.mean_touch_recall,
            analysis.mean_edit_precision,
            analysis.mean_edit_recall,
        ]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Rate')
        ax.set_title('Localization Accuracy')
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 2. Localization: Resolved vs Unresolved
        ax = axes[0, 1]
        categories = ['Resolved', 'Unresolved']
        recalls = [analysis.resolved_mean_touch_recall, analysis.unresolved_mean_touch_recall]
        ax.bar(categories, recalls, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Touch Recall')
        ax.set_title('Localization: Resolved vs Unresolved')
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', alpha=0.3)
        for i, val in enumerate(recalls):
            ax.text(i, val, f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 3. Trajectory Phases
        ax = axes[0, 2]
        phases = ['Exploration', 'Editing', 'Verification']
        phase_values = [
            analysis.mean_exploration_steps,
            analysis.mean_editing_steps,
            analysis.mean_verification_steps,
        ]
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        bars = ax.bar(phases, phase_values, color=colors, alpha=0.7)
        ax.set_ylabel('Steps')
        ax.set_title('Average Trajectory Phases')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, phase_values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Time to First Edit: Resolved vs Unresolved
        ax = axes[1, 0]
        categories = ['Resolved', 'Unresolved']
        first_edits = [
            analysis.resolved_mean_steps_to_first_edit,
            analysis.unresolved_mean_steps_to_first_edit,
        ]
        ax.bar(categories, first_edits, color=['#2ecc71', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Steps to First Edit')
        ax.set_title('Time to First Edit')
        ax.grid(axis='y', alpha=0.3)
        for i, val in enumerate(first_edits):
            ax.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        # 5. Error Analysis
        ax = axes[1, 1]
        error_metrics = ['Mean\nError Rate', 'Recovery\nRate']
        error_values = [analysis.mean_error_rate, analysis.overall_recovery_rate]
        colors = ['#e74c3c', '#2ecc71']
        bars = ax.bar(error_metrics, error_values, color=colors, alpha=0.7)
        ax.set_ylabel('Rate')
        ax.set_title('Error & Recovery Rates')
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, error_values):
            ax.text(bar.get_x() + bar.get_width()/2., val,
                   f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 6. Summary Statistics
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
        Trajectory Dynamics Summary
        ═══════════════════════════════════
        
        Localization:
          • Mean Touch Precision: {analysis.mean_touch_precision:.1%}
          • Mean Touch Recall: {analysis.mean_touch_recall:.1%}
          • Resolved Touch Recall: {analysis.resolved_mean_touch_recall:.1%}
          • Unresolved Touch Recall: {analysis.unresolved_mean_touch_recall:.1%}
        
        Phases:
          • Mean Exploration Steps: {analysis.mean_exploration_steps:.1f}
          • Mean Editing Steps: {analysis.mean_editing_steps:.1f}
          • Mean Verification Steps: {analysis.mean_verification_steps:.1f}
          • Mean Steps to First Edit: {analysis.mean_steps_to_first_edit:.1f}
        
        Errors:
          • Mean Error Rate: {analysis.mean_error_rate:.1%}
          • Overall Recovery Rate: {analysis.overall_recovery_rate:.1%}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / f'trajectory_dynamics_{self._safe_name(analysis.run_name)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved trajectory dynamics plot to {output_path}")
        return output_path
    
    # ==================== Sequence Pattern Plots ====================
    
    def plot_sequence_patterns_all(
        self,
        runs: list[Run],
    ) -> list[Path]:
        """Generate sequence pattern plots for all runs.
        
        Args:
            runs: List of runs to analyze
            
        Returns:
            List of generated plot file paths
        """
        analyzer = TrajectoryDynamicsAnalyzer()
        
        generated: list[Path] = []
        for run in runs:
            analysis = analyzer.analyze_run(run)
            path = self.plot_sequence_patterns(analysis)
            if path:
                generated.append(path)
        
        return generated
    
    def plot_sequence_patterns(
        self,
        analysis: TrajectoryDynamicsAnalysis,
    ) -> Path | None:
        """Plot tool call sequence patterns.
        
        Args:
            analysis: TrajectoryDynamicsAnalysis object
            
        Returns:
            Path to generated plot or None
        """
        patterns = analysis.sequence_patterns
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle(
            f'Tool Sequence Patterns: {analysis.run_name}',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Top Success Bigrams vs Failure Bigrams
        ax = axes[0, 0]
        success_bigrams = patterns.success_bigrams.most_common(15)
        failure_bigrams = patterns.failure_bigrams.most_common(15)
        
        if success_bigrams or failure_bigrams:
            # Combine and get top patterns
            all_bigrams = set(b[0] for b in success_bigrams) | set(b[0] for b in failure_bigrams)
            top_bigrams = sorted(
                all_bigrams,
                key=lambda b: patterns.success_bigrams.get(b, 0) + patterns.failure_bigrams.get(b, 0),
                reverse=True
            )[:12]
            
            if top_bigrams:
                success_counts = [patterns.success_bigrams.get(b, 0) for b in top_bigrams]
                failure_counts = [patterns.failure_bigrams.get(b, 0) for b in top_bigrams]
                
                x = np.arange(len(top_bigrams))
                width = 0.35
                
                ax.bar(x - width/2, success_counts, width, label='Resolved', color='#2ecc71', alpha=0.7)
                ax.bar(x + width/2, failure_counts, width, label='Unresolved', color='#e74c3c', alpha=0.7)
                
                labels = [f'{b[0]}→{b[1]}' for b in top_bigrams]
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Count')
                ax.set_title('Tool Bigrams: Resolved vs Unresolved')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # 2. Distinctive Patterns (Success-only vs Failure-only)
        ax = axes[0, 1]
        success_only = patterns.success_only_patterns[:8]
        failure_only = patterns.failure_only_patterns[:8]
        
        labels = []
        values = []
        colors = []
        
        for pattern, count in success_only:
            labels.append(f'{pattern[0]}→{pattern[1]}')
            values.append(count)
            colors.append('#2ecc71')
        
        for pattern, count in failure_only:
            labels.append(f'{pattern[0]}→{pattern[1]}')
            values.append(-count)  # Negative for failure
            colors.append('#e74c3c')
        
        if labels:
            y_pos = np.arange(len(labels))
            ax.barh(y_pos, values, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=8)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel('Count (Left=Failure Only, Right=Success Only)')
            ax.set_title('Distinctive Patterns')
            ax.grid(axis='x', alpha=0.3)
        
        # 3. Trajectory Start Patterns
        ax = axes[1, 0]
        success_starts = patterns.success_starts.most_common(8)
        failure_starts = patterns.failure_starts.most_common(8)
        
        if success_starts or failure_starts:
            all_starts = set(s[0] for s in success_starts) | set(s[0] for s in failure_starts)
            top_starts = sorted(
                all_starts,
                key=lambda s: patterns.success_starts.get(s, 0) + patterns.failure_starts.get(s, 0),
                reverse=True
            )[:8]
            
            if top_starts:
                success_counts = [patterns.success_starts.get(s, 0) for s in top_starts]
                failure_counts = [patterns.failure_starts.get(s, 0) for s in top_starts]
                
                x = np.arange(len(top_starts))
                width = 0.35
                
                ax.bar(x - width/2, success_counts, width, label='Resolved', color='#2ecc71', alpha=0.7)
                ax.bar(x + width/2, failure_counts, width, label='Unresolved', color='#e74c3c', alpha=0.7)
                
                labels = ['→'.join(s[:2]) + '...' if len(s) > 2 else '→'.join(s) for s in top_starts]
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Count')
                ax.set_title('First 3 Tools: Resolved vs Unresolved')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # 4. Trajectory End Patterns
        ax = axes[1, 1]
        success_ends = patterns.success_ends.most_common(8)
        failure_ends = patterns.failure_ends.most_common(8)
        
        if success_ends or failure_ends:
            all_ends = set(e[0] for e in success_ends) | set(e[0] for e in failure_ends)
            top_ends = sorted(
                all_ends,
                key=lambda e: patterns.success_ends.get(e, 0) + patterns.failure_ends.get(e, 0),
                reverse=True
            )[:8]
            
            if top_ends:
                success_counts = [patterns.success_ends.get(e, 0) for e in top_ends]
                failure_counts = [patterns.failure_ends.get(e, 0) for e in top_ends]
                
                x = np.arange(len(top_ends))
                width = 0.35
                
                ax.bar(x - width/2, success_counts, width, label='Resolved', color='#2ecc71', alpha=0.7)
                ax.bar(x + width/2, failure_counts, width, label='Unresolved', color='#e74c3c', alpha=0.7)
                
                labels = ['...→' + '→'.join(e[-2:]) if len(e) > 2 else '→'.join(e) for e in top_ends]
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Count')
                ax.set_title('Last 3 Tools: Resolved vs Unresolved')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / f'sequence_patterns_{self._safe_name(analysis.run_name)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sequence patterns plot to {output_path}")
        return output_path
    
    # ==================== Failure Taxonomy Plots ====================
    
    def plot_failure_taxonomy_all(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[Path]:
        """Generate failure taxonomy plots for all runs.
        
        Args:
            runs: List of runs to analyze
            instances: Optional instances dict
            
        Returns:
            List of generated plot file paths
        """
        analyzer = FailureTaxonomyAnalyzer()
        taxonomies = analyzer.analyze_runs(runs, instances)
        
        generated: list[Path] = []
        for taxonomy in taxonomies:
            path = self.plot_failure_taxonomy(taxonomy)
            if path:
                generated.append(path)
        
        return generated
    
    def plot_failure_taxonomy(
        self,
        taxonomy: FailureTaxonomy,
    ) -> Path | None:
        """Plot failure taxonomy analysis.
        
        Args:
            taxonomy: FailureTaxonomy object
            
        Returns:
            Path to generated plot or None
        """
        if taxonomy.unresolved_count == 0:
            logger.info(f"No failures to analyze for {taxonomy.run_name}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            f'Failure Taxonomy: {taxonomy.run_name}\n'
            f'({taxonomy.unresolved_count}/{taxonomy.total_trajectories} unresolved)',
            fontsize=16, fontweight='bold'
        )
        
        # 1. Primary Failure Categories (Pie Chart)
        ax = axes[0, 0]
        if taxonomy.most_common_primary:
            categories = [c[0].replace('_', ' ').title() for c in taxonomy.most_common_primary]
            counts = [c[1] for c in taxonomy.most_common_primary]
            
            # Color mapping
            color_map = {
                'Wrong Files': '#e74c3c',
                'Missed Files': '#c0392b',
                'No Localization': '#922b21',
                'No Edit Attempted': '#3498db',
                'Edit Not Applied': '#2980b9',
                'Wrong Edit': '#1f618d',
                'Max Steps Reached': '#f39c12',
                'Stuck In Loop': '#d68910',
                'Cascade Errors': '#9b59b6',
                'Patch Breaks Tests': '#27ae60',
                'Unknown': '#95a5a6',
            }
            colors = [color_map.get(c, '#95a5a6') for c in categories]
            
            wedges, texts, autotexts = ax.pie(
                counts, labels=categories, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 9}
            )
            ax.set_title('Primary Failure Categories')
        
        # 2. Failure Categories Bar Chart
        ax = axes[0, 1]
        if taxonomy.most_common_primary:
            categories = [c[0].replace('_', ' ').title() for c in taxonomy.most_common_primary[:10]]
            counts = [c[1] for c in taxonomy.most_common_primary[:10]]
            
            y_pos = np.arange(len(categories))
            bars = ax.barh(y_pos, counts, color='#e74c3c', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.invert_yaxis()
            ax.set_xlabel('Count')
            ax.set_title('Top Primary Failure Categories')
            ax.grid(axis='x', alpha=0.3)
            
            for bar, count in zip(bars, counts):
                pct = count / taxonomy.unresolved_count * 100
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                       f'{count} ({pct:.1f}%)', va='center', fontsize=9)
        
        # 3. Failure Type Breakdown
        ax = axes[1, 0]
        type_rates = [
            ('Localization', taxonomy.localization_failure_rate),
            ('Edit', taxonomy.edit_failure_rate),
            ('Process', taxonomy.process_failure_rate),
        ]
        
        labels = [t[0] for t in type_rates]
        rates = [t[1] for t in type_rates]
        colors = ['#e74c3c', '#3498db', '#f39c12']
        
        bars = ax.bar(labels, rates, color=colors, alpha=0.7)
        ax.set_ylabel('Failure Rate')
        ax.set_title('Failure Type Breakdown')
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(axis='y', alpha=0.3)
        
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2., rate,
                   f'{rate:.1%}', ha='center', va='bottom', fontsize=10)
        
        # 4. Secondary Factors & Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Build summary text
        secondary_text = ""
        if taxonomy.most_common_secondary:
            secondary_text = "\n        Secondary Factors:\n"
            for cat, count in taxonomy.most_common_secondary[:5]:
                secondary_text += f"          • {cat.replace('_', ' ').title()}: {count}\n"
        
        summary_text = f"""
        Failure Analysis Summary
        ═══════════════════════════════════
        
        Total Trajectories: {taxonomy.total_trajectories}
        Resolved: {taxonomy.resolved_count} ({taxonomy.resolved_count/taxonomy.total_trajectories:.1%})
        Unresolved: {taxonomy.unresolved_count} ({taxonomy.unresolved_count/taxonomy.total_trajectories:.1%})
        
        Failure Type Breakdown:
          • Localization Issues: {taxonomy.localization_failure_rate:.1%}
          • Edit Issues: {taxonomy.edit_failure_rate:.1%}
          • Process Issues: {taxonomy.process_failure_rate:.1%}
        
        Contributing Issues:
          • Loops Leading to Failure: {taxonomy.loops_leading_to_failure}
          • Error Cascades: {taxonomy.errors_leading_to_failure}
        {secondary_text}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = self.output_dir / f'failure_taxonomy_{self._safe_name(taxonomy.run_name)}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved failure taxonomy plot to {output_path}")
        return output_path

