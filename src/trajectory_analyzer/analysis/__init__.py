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
]

