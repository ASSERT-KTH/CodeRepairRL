"""Analysis modules for trajectory analysis."""

from .metrics import MetricsExtractor, RunMetrics
from .comparator import RunComparator
from .transfer import (
    TransferAnalyzer,
    DeepTransferAnalysis,
    McNemarResult,
    ErrorModeAnalysis,
    ToolVocabularyAnalysis,
    ErrorCategory,
)
from .variance import (
    VarianceAnalyzer, VarianceAnalysis, InstanceVarianceMetrics,
    DeepVarianceAnalysis, RunMetricsStatistics,
)
from .dynamics import (
    TrajectoryDynamicsAnalyzer, TrajectoryDynamicsAnalysis,
    LocalizationMetrics, PhaseMetrics, ErrorRecoveryMetrics, SequencePatterns,
)
from .failure import (
    FailureTaxonomyAnalyzer, FailureTaxonomy, FailureDiagnosis, FailureCategory,
)

__all__ = [
    "MetricsExtractor",
    "RunMetrics",
    "RunComparator",
    "TransferAnalyzer",
    "DeepTransferAnalysis",
    "McNemarResult",
    "ErrorModeAnalysis",
    "ToolVocabularyAnalysis",
    "ErrorCategory",
    "VarianceAnalyzer",
    "VarianceAnalysis",
    "InstanceVarianceMetrics",
    "DeepVarianceAnalysis",
    "RunMetricsStatistics",
    "TrajectoryDynamicsAnalyzer",
    "TrajectoryDynamicsAnalysis",
    "LocalizationMetrics",
    "PhaseMetrics",
    "ErrorRecoveryMetrics",
    "SequencePatterns",
    "FailureTaxonomyAnalyzer",
    "FailureTaxonomy",
    "FailureDiagnosis",
    "FailureCategory",
]

