"""Hydra CLI entry point for trajectory analyzer."""

import logging
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


if __name__ == "__main__":
    main()

