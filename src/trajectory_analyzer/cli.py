"""Hydra CLI entry point for trajectory analyzer."""

import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .loaders import ClaudeCodeLoader, NanoAgentLoader, R2EGymLoader, SWEAgentLoader, TrajectoryLoader
from .models import Run
from .analysis import MetricsExtractor, RunComparator
from .plotting import TrajectoryPlotter

logger = logging.getLogger(__name__)


def get_loader(format: str) -> TrajectoryLoader:
    """Get the appropriate loader for the format.
    
    Args:
        format: One of 'claude_code', 'nano_agent', 'r2e_gym', 'swe_agent'
        
    Returns:
        TrajectoryLoader instance
    """
    loaders = {
        'claude_code': ClaudeCodeLoader,
        'nano_agent': NanoAgentLoader,
        'r2e_gym': R2EGymLoader,
        'swe_agent': SWEAgentLoader,
    }
    
    if format not in loaders:
        raise ValueError(f"Unknown format: {format}. Choose from {list(loaders.keys())}")
    
    return loaders[format]()


def load_run_from_config(run_config: DictConfig) -> Run:
    """Load a run from configuration.
    
    Args:
        run_config: Run configuration
        
    Returns:
        Run object with loaded trajectories
    """
    loader = get_loader(run_config.format)
    
    return loader.load_run(
        name=run_config.name,
        scaffold=run_config.scaffold,
        base_model=run_config.base_model,
        trajectories_path=run_config.trajectories,
        results_path=run_config.get('results'),
        lora_adapter=run_config.get('lora_adapter'),
    )


def print_summary(runs: list[Run], comparator: RunComparator) -> None:
    """Print summary statistics for all runs."""
    print("\n" + "=" * 70)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("=" * 70)
    
    for run in runs:
        print(f"\n{run.name}")
        print("-" * len(run.name))
        print(f"  Scaffold: {run.scaffold}")
        print(f"  Base Model: {run.base_model}")
        if run.lora_adapter:
            print(f"  LoRA Adapter: {run.lora_adapter}")
        print(f"  Instances: {run.num_instances}")
        print(f"  Resolved: {run.num_resolved} ({run.resolve_rate:.1%})")
        print(f"  With Patch: {run.num_with_patch} ({run.patch_rate:.1%})")
        print(f"  Avg Tool Calls: {run.get_avg_tool_calls():.1f}")
        print(f"  Avg Tokens: {run.get_avg_tokens():.0f}")
        print(f"  Avg Steps: {run.get_avg_steps():.1f}")
        
        # Top tools
        tool_counts = run.get_total_tool_counts()
        if tool_counts:
            sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top Tools: {', '.join(f'{t}({c})' for t, c in sorted_tools)}")
    
    if len(runs) > 1:
        summary = comparator.get_comparison_summary(runs)
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)
        print(f"  Best Run: {summary['best_run']}")
        print(f"  Worst Run: {summary['worst_run']}")
        print(f"  Average Resolution Rate: {summary['avg_resolve_rate']:.1%}")
    
    print()


def generate_swebench_score_table(runs: list[Run], output_dir: Path) -> Path:
    """Generate a markdown table with models as rows, scaffolds as columns, and SWE-bench scores.
    
    For runs with multiple entries (same model+scaffold), compute mean ± stddev.
    
    Args:
        runs: List of runs to analyze
        output_dir: Directory to save the markdown file
        
    Returns:
        Path to the generated markdown file
    """
    # Group runs by (base_model, scaffold)
    grouped: dict[tuple[str, str], list[Run]] = defaultdict(list)
    for run in runs:
        key = (run.base_model, run.scaffold)
        grouped[key].append(run)
    
    # Collect all unique models and scaffolds
    all_models = sorted(set(run.base_model for run in runs))
    all_scaffolds = sorted(set(run.scaffold for run in runs))
    
    # Build the table data
    table_data: dict[str, dict[str, str]] = {}
    
    for model in all_models:
        table_data[model] = {}
        for scaffold in all_scaffolds:
            key = (model, scaffold)
            matching_runs = grouped.get(key, [])
            
            if not matching_runs:
                table_data[model][scaffold] = "-"
            elif len(matching_runs) == 1:
                # Single run: just show the score
                score = matching_runs[0].resolve_rate
                table_data[model][scaffold] = f"{score:.1%}"
            else:
                # Multiple runs: compute mean ± stddev
                scores = [run.resolve_rate for run in matching_runs]
                mean_score = statistics.mean(scores)
                if len(scores) > 1:
                    stddev_score = statistics.stdev(scores)
                    table_data[model][scaffold] = f"{mean_score:.1%} ± {stddev_score:.1%}"
                else:
                    table_data[model][scaffold] = f"{mean_score:.1%}"
    
    # Generate markdown table
    lines = ["# SWE-bench Scores by Model and Scaffold\n"]
    lines.append("This table shows the resolution rate (SWE-bench score) for each model-scaffold combination.\n")
    lines.append("For multiple runs with the same model and scaffold, the mean ± standard deviation is shown.\n")
    lines.append("")
    
    # Table header
    header = "| Model | " + " | ".join(all_scaffolds) + " |"
    lines.append(header)
    separator = "|" + "|".join(["---"] * (len(all_scaffolds) + 1)) + "|"
    lines.append(separator)
    
    # Table rows
    for model in all_models:
        row = f"| {model} |"
        for scaffold in all_scaffolds:
            cell_value = table_data[model].get(scaffold, "-")
            row += f" {cell_value} |"
        lines.append(row)
    
    # Add summary statistics
    lines.append("")
    lines.append("## Summary Statistics\n")
    lines.append(f"- Total runs: {len(runs)}")
    lines.append(f"- Unique models: {len(all_models)}")
    lines.append(f"- Unique scaffolds: {len(all_scaffolds)}")
    lines.append("")
    
    # List runs with multiple entries
    multi_run_combinations = [
        (model, scaffold, len(runs_list))
        for (model, scaffold), runs_list in grouped.items()
        if len(runs_list) > 1
    ]
    
    if multi_run_combinations:
        lines.append("## Multiple Runs (Mean ± StdDev Computed)\n")
        for model, scaffold, count in sorted(multi_run_combinations):
            scores = [run.resolve_rate for run in grouped[(model, scaffold)]]
            mean_score = statistics.mean(scores)
            stddev_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
            lines.append(f"- **{model}** on **{scaffold}**: {count} runs → {mean_score:.1%} ± {stddev_score:.1%}")
        lines.append("")
    
    # Write to file
    output_path = output_dir / "swebench_scores_table.md"
    output_path.write_text("\n".join(lines))
    logger.info(f"Generated SWE-bench scores table at {output_path}")
    
    return output_path


@hydra.main(version_base=None, config_path="conf", config_name="analyzer_config")
def main(cfg: DictConfig) -> None:
    """Main entry point for trajectory analyzer.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Trajectory Analyzer starting...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load runs
    runs: list[Run] = []
    for run_config in cfg.runs:
        try:
            run = load_run_from_config(run_config)
            runs.append(run)
            logger.info(f"Loaded run '{run.name}' with {len(run)} trajectories")
        except Exception as e:
            logger.error(f"Failed to load run '{run_config.name}': {e}")
    
    if not runs:
        logger.error("No runs loaded. Check your configuration.")
        return
    
    # Initialize analysis components
    metrics_extractor = MetricsExtractor()
    comparator = RunComparator()
    plotter = TrajectoryPlotter(output_dir=cfg.output_dir)
    
    # Print summary
    print_summary(runs, comparator)
    
    # Generate plots
    plots_to_generate = list(cfg.plots) if cfg.plots else None
    
    logger.info(f"Generating plots: {plots_to_generate or 'all'}")
    
    # Set model to trained scaffold mapping for transfer analysis
    model_to_scaffold = dict(cfg.get('model_to_trained_scaffold', {}))
    
    generated_plots = plotter.plot_all(
        runs, 
        plots_to_generate,
        model_to_trained_scaffold=model_to_scaffold,
    )
    
    logger.info(f"Generated {len(generated_plots)} plots in {cfg.output_dir}")
    print(f"\nGenerated {len(generated_plots)} plots in {cfg.output_dir}")
    
    # Generate SWE-bench scores table
    output_dir_path = Path(cfg.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    table_path = generate_swebench_score_table(runs, output_dir_path)
    print(f"Generated SWE-bench scores table at {table_path}")


if __name__ == "__main__":
    main()

