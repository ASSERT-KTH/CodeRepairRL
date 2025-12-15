"""Cross-run comparison and transfer learning analysis."""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any

from ..models import Run
from .metrics import MetricsExtractor, RunMetrics


@dataclass
class TransferAnalysis:
    """Analysis of transfer learning between scaffolds."""
    
    base_model: str
    """Base model being compared."""
    
    source_scaffold: str
    """Scaffold the model was trained on."""
    
    target_scaffold: str
    """Scaffold the model is evaluated on."""
    
    source_run: str
    """Run name for source scaffold."""
    
    target_run: str
    """Run name for target scaffold."""
    
    # Performance metrics
    source_resolve_rate: float = 0.0
    target_resolve_rate: float = 0.0
    transfer_delta: float = 0.0  # target - source (negative = degradation)
    
    # Tool usage comparison
    source_avg_tool_calls: float = 0.0
    target_avg_tool_calls: float = 0.0
    tool_usage_delta: float = 0.0
    
    # Tool distribution similarity (Jaccard index of top tools)
    tool_overlap: float = 0.0
    
    @property
    def transfer_successful(self) -> bool:
        """Whether transfer maintained or improved performance."""
        return self.transfer_delta >= 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_model': self.base_model,
            'source_scaffold': self.source_scaffold,
            'target_scaffold': self.target_scaffold,
            'source_run': self.source_run,
            'target_run': self.target_run,
            'source_resolve_rate': self.source_resolve_rate,
            'target_resolve_rate': self.target_resolve_rate,
            'transfer_delta': self.transfer_delta,
            'source_avg_tool_calls': self.source_avg_tool_calls,
            'target_avg_tool_calls': self.target_avg_tool_calls,
            'tool_usage_delta': self.tool_usage_delta,
            'tool_overlap': self.tool_overlap,
            'transfer_successful': self.transfer_successful,
        }


@dataclass
class FinetuningAnalysis:
    """Analysis of fine-tuning impact."""
    
    base_model: str
    scaffold: str
    lora_adapter: str
    
    base_run: str
    finetuned_run: str
    
    # Performance comparison
    base_resolve_rate: float = 0.0
    finetuned_resolve_rate: float = 0.0
    improvement: float = 0.0  # finetuned - base
    
    # Tool usage comparison
    base_avg_tool_calls: float = 0.0
    finetuned_avg_tool_calls: float = 0.0
    
    # Changes in tool distribution
    new_tools: list[str] = field(default_factory=list)  # Tools used only after finetuning
    dropped_tools: list[str] = field(default_factory=list)  # Tools no longer used
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_model': self.base_model,
            'scaffold': self.scaffold,
            'lora_adapter': self.lora_adapter,
            'base_run': self.base_run,
            'finetuned_run': self.finetuned_run,
            'base_resolve_rate': self.base_resolve_rate,
            'finetuned_resolve_rate': self.finetuned_resolve_rate,
            'improvement': self.improvement,
            'base_avg_tool_calls': self.base_avg_tool_calls,
            'finetuned_avg_tool_calls': self.finetuned_avg_tool_calls,
            'new_tools': self.new_tools,
            'dropped_tools': self.dropped_tools,
        }


class RunComparator:
    """Compare runs for transfer learning and fine-tuning analysis."""
    
    def __init__(self):
        self.metrics_extractor = MetricsExtractor()
    
    def compare_runs(self, runs: list[Run]) -> dict[str, RunMetrics]:
        """Extract and compare metrics across multiple runs.
        
        Args:
            runs: List of runs to compare
            
        Returns:
            Dictionary mapping run names to their metrics
        """
        return {
            run.name: self.metrics_extractor.extract_run_metrics(run)
            for run in runs
        }
    
    def analyze_transfer(
        self,
        source_run: Run,
        target_run: Run,
        source_scaffold: str,
    ) -> TransferAnalysis:
        """Analyze transfer learning from one scaffold to another.
        
        Args:
            source_run: Run on the scaffold the model was trained for
            target_run: Run on a different scaffold
            source_scaffold: Name of the source scaffold
            
        Returns:
            TransferAnalysis with comparison results
        """
        source_metrics = self.metrics_extractor.extract_run_metrics(source_run)
        target_metrics = self.metrics_extractor.extract_run_metrics(target_run)
        
        # Compute tool overlap (Jaccard index)
        source_tools = set(source_metrics.tool_counts.keys())
        target_tools = set(target_metrics.tool_counts.keys())
        
        if source_tools or target_tools:
            intersection = len(source_tools & target_tools)
            union = len(source_tools | target_tools)
            tool_overlap = intersection / union if union > 0 else 0.0
        else:
            tool_overlap = 0.0
        
        return TransferAnalysis(
            base_model=source_run.base_model,
            source_scaffold=source_scaffold,
            target_scaffold=target_run.scaffold,
            source_run=source_run.name,
            target_run=target_run.name,
            source_resolve_rate=source_metrics.resolve_rate,
            target_resolve_rate=target_metrics.resolve_rate,
            transfer_delta=target_metrics.resolve_rate - source_metrics.resolve_rate,
            source_avg_tool_calls=source_metrics.avg_tool_calls,
            target_avg_tool_calls=target_metrics.avg_tool_calls,
            tool_usage_delta=target_metrics.avg_tool_calls - source_metrics.avg_tool_calls,
            tool_overlap=tool_overlap,
        )
    
    def analyze_finetuning(
        self,
        base_run: Run,
        finetuned_run: Run,
    ) -> FinetuningAnalysis:
        """Analyze impact of fine-tuning.
        
        Args:
            base_run: Run with base model
            finetuned_run: Run with fine-tuned model
            
        Returns:
            FinetuningAnalysis with comparison results
        """
        base_metrics = self.metrics_extractor.extract_run_metrics(base_run)
        ft_metrics = self.metrics_extractor.extract_run_metrics(finetuned_run)
        
        # Find tool changes
        base_tools = set(base_metrics.tool_counts.keys())
        ft_tools = set(ft_metrics.tool_counts.keys())
        
        new_tools = list(ft_tools - base_tools)
        dropped_tools = list(base_tools - ft_tools)
        
        return FinetuningAnalysis(
            base_model=base_run.base_model,
            scaffold=base_run.scaffold,
            lora_adapter=finetuned_run.lora_adapter or "unknown",
            base_run=base_run.name,
            finetuned_run=finetuned_run.name,
            base_resolve_rate=base_metrics.resolve_rate,
            finetuned_resolve_rate=ft_metrics.resolve_rate,
            improvement=ft_metrics.resolve_rate - base_metrics.resolve_rate,
            base_avg_tool_calls=base_metrics.avg_tool_calls,
            finetuned_avg_tool_calls=ft_metrics.avg_tool_calls,
            new_tools=new_tools,
            dropped_tools=dropped_tools,
        )
    
    def find_transfer_pairs(
        self,
        runs: list[Run],
        model_to_trained_scaffold: dict[str, str],
    ) -> list[TransferAnalysis]:
        """Find and analyze all transfer learning pairs.
        
        Args:
            runs: All available runs
            model_to_trained_scaffold: Mapping of base models to their training scaffold
                e.g., {"agentica-org/DeepSWE-Preview": "r2e-gym"}
            
        Returns:
            List of TransferAnalysis for each pair
        """
        analyses = []
        
        # Group runs by base model
        runs_by_model: dict[str, list[Run]] = defaultdict(list)
        for run in runs:
            runs_by_model[run.base_model].append(run)
        
        # For each model, compare training scaffold to other scaffolds
        for model, model_runs in runs_by_model.items():
            trained_scaffold = model_to_trained_scaffold.get(model)
            if not trained_scaffold:
                continue
            
            # Find source run (on training scaffold)
            source_runs = [r for r in model_runs if r.scaffold == trained_scaffold]
            if not source_runs:
                continue
            
            source_run = source_runs[0]  # Take first if multiple
            
            # Compare to runs on other scaffolds
            for target_run in model_runs:
                if target_run.scaffold != trained_scaffold:
                    analysis = self.analyze_transfer(
                        source_run, target_run, trained_scaffold
                    )
                    analyses.append(analysis)
        
        return analyses
    
    def find_finetuning_pairs(self, runs: list[Run]) -> list[FinetuningAnalysis]:
        """Find and analyze all base vs fine-tuned pairs.
        
        Args:
            runs: All available runs
            
        Returns:
            List of FinetuningAnalysis for each pair
        """
        analyses = []
        
        # Group runs by (base_model, scaffold)
        runs_by_key: dict[tuple[str, str], list[Run]] = defaultdict(list)
        for run in runs:
            key = (run.base_model, run.scaffold)
            runs_by_key[key].append(run)
        
        # For each group, compare base to fine-tuned versions
        for (model, scaffold), group_runs in runs_by_key.items():
            base_runs = [r for r in group_runs if not r.is_finetuned]
            ft_runs = [r for r in group_runs if r.is_finetuned]
            
            if not base_runs:
                continue
            
            base_run = base_runs[0]  # Take first if multiple
            
            for ft_run in ft_runs:
                analysis = self.analyze_finetuning(base_run, ft_run)
                analyses.append(analysis)
        
        return analyses
    
    def get_comparison_summary(
        self,
        runs: list[Run],
    ) -> dict[str, Any]:
        """Get a summary of all run comparisons.
        
        Args:
            runs: All available runs
            
        Returns:
            Summary dictionary with metrics and comparisons
        """
        metrics = self.compare_runs(runs)
        
        # Sort by resolve rate
        sorted_runs = sorted(
            metrics.items(),
            key=lambda x: x[1].resolve_rate,
            reverse=True
        )
        
        return {
            'runs': {name: m.to_dict() for name, m in sorted_runs},
            'best_run': sorted_runs[0][0] if sorted_runs else None,
            'worst_run': sorted_runs[-1][0] if sorted_runs else None,
            'avg_resolve_rate': (
                sum(m.resolve_rate for m in metrics.values()) / len(metrics)
                if metrics else 0.0
            ),
        }

