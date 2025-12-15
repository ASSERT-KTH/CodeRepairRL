"""Deep transfer learning analysis.

Provides detailed analysis of transfer learning between scaffolds:
1. Instance-level correlation with McNemar's test
2. Error mode analysis and categorization
3. Tool vocabulary alignment analysis
"""

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Any
from enum import Enum
import math

from ..models import Run, Trajectory, ToolCall
import re


# Common error patterns to look for in tool results
ERROR_PATTERNS = [
    (r'command failed with exit code (\d+)', 'exit_code_{0}'),
    (r'No such file or directory', 'file_not_found'),
    (r'Permission denied', 'permission_denied'),
    (r'command not found', 'command_not_found'),
    (r'FileNotFoundError', 'file_not_found'),
    (r'ModuleNotFoundError', 'module_not_found'),
    (r'ImportError', 'import_error'),
    (r'SyntaxError', 'syntax_error'),
    (r'IndentationError', 'indentation_error'),
    (r'NameError', 'name_error'),
    (r'TypeError', 'type_error'),
    (r'ValueError', 'value_error'),
    (r'AttributeError', 'attribute_error'),
    (r'KeyError', 'key_error'),
    (r'IndexError', 'index_error'),
    (r'RuntimeError', 'runtime_error'),
    (r'AssertionError', 'assertion_error'),
    (r'OSError', 'os_error'),
    (r'IOError', 'io_error'),
    (r'Traceback \(most recent call last\)', 'python_traceback'),
    (r'FAILED', 'test_failed'),
    (r'ERROR', 'generic_error'),
    (r'fatal:', 'git_fatal'),
    (r'error:', 'generic_error'),
    (r'not found', 'not_found'),
    (r'does not exist', 'does_not_exist'),
    (r'invalid', 'invalid_input'),
    (r'timeout', 'timeout'),
    (r'connection refused', 'connection_refused'),
    (r'patch.*failed', 'patch_failed'),
    (r'could not apply', 'patch_failed'),
    (r'SEARCH.*not found', 'search_text_not_found'),
    (r'multiple matches', 'multiple_matches'),
    (r'No changes made', 'no_changes'),
]


def extract_error_pattern(result: str, mode: str = "categorized") -> str:
    """Extract error pattern from a tool result string.
    
    Args:
        result: The tool result/output string
        mode: "categorized" for predefined patterns, "raw" for first N words
        
    Returns:
        Identified error pattern or 'unknown_error'
    """
    if not result:
        return 'empty_result'
    
    if mode == "raw":
        return extract_raw_error_prefix(result)
    
    result_lower = result.lower()
    
    for pattern, label in ERROR_PATTERNS:
        match = re.search(pattern, result, re.IGNORECASE)
        if match:
            # Handle patterns with capture groups
            if '{0}' in label and match.groups():
                return label.format(match.group(1))
            return label
    
    # If no pattern matched but marked as error, classify by length
    if len(result) < 50:
        return f'short_error: {result[:30]}'
    
    return 'unknown_error'


def extract_raw_error_prefix(result: str, num_words: int = 5) -> str:
    """Extract first N words from error result as category.
    
    Args:
        result: The tool result/output string
        num_words: Number of words to extract
        
    Returns:
        First N words of the error, cleaned up
    """
    if not result:
        return 'empty_result'
    
    # Clean up the result - remove newlines, extra spaces
    cleaned = ' '.join(result.split())
    
    # Get first N words
    words = cleaned.split()[:num_words]
    
    if not words:
        return 'empty_result'
    
    # Join and truncate if too long
    prefix = ' '.join(words)
    if len(prefix) > 60:
        prefix = prefix[:57] + '...'
    
    return prefix


def format_tool_call_signature(tool_name: str, arguments: dict, max_len: int = 80) -> str:
    """Format a tool call with its arguments into a readable signature.
    
    Args:
        tool_name: Name of the tool
        arguments: Tool arguments dictionary
        max_len: Maximum length of the signature
        
    Returns:
        Formatted signature like "shell(cmd='grep -r ...')"
    """
    if not arguments:
        return f"{tool_name}()"
    
    # Format key arguments
    arg_parts = []
    for key, value in arguments.items():
        if value is None:
            continue
        
        # Convert value to string and truncate if needed
        str_val = str(value)
        if len(str_val) > 30:
            str_val = str_val[:27] + '...'
        
        # Clean up whitespace
        str_val = ' '.join(str_val.split())
        
        arg_parts.append(f"{key}='{str_val}'")
    
    if not arg_parts:
        return f"{tool_name}()"
    
    # Join arguments
    args_str = ', '.join(arg_parts)
    signature = f"{tool_name}({args_str})"
    
    # Truncate if too long
    if len(signature) > max_len:
        signature = signature[:max_len-3] + '...'
    
    return signature


class ErrorCategory(Enum):
    """Categories of errors in agent trajectories."""
    
    # Tool-level errors
    TOOL_NOT_FOUND = "tool_not_found"  # Called a tool that doesn't exist
    TOOL_PARSE_ERROR = "tool_parse_error"  # Tool call couldn't be parsed
    TOOL_WRONG_ARGS = "tool_wrong_args"  # Correct tool, but arguments failed
    TOOL_EXECUTION_ERROR = "tool_execution_error"  # Tool ran but returned error
    
    # Strategy-level errors
    NO_PATCH_GENERATED = "no_patch_generated"  # Never produced a patch
    PATCH_FAILED = "patch_failed"  # Patch didn't apply or was wrong
    MAX_STEPS_REACHED = "max_steps_reached"  # Hit step limit
    
    # Other
    UNKNOWN = "unknown"


@dataclass
class InstanceTransferResult:
    """Result of transfer for a single instance."""
    
    instance_id: str
    source_resolved: bool
    target_resolved: bool
    
    @property
    def transfer_category(self) -> str:
        """Categorize the transfer outcome."""
        if self.source_resolved and self.target_resolved:
            return "both_success"
        elif self.source_resolved and not self.target_resolved:
            return "transfer_loss"  # Worked on source, failed on target
        elif not self.source_resolved and self.target_resolved:
            return "transfer_gain"  # Failed on source, worked on target
        else:
            return "both_failure"


@dataclass
class McNemarResult:
    """Result of McNemar's test for paired binary outcomes."""
    
    # Contingency table counts
    both_success: int = 0  # Resolved on both scaffolds
    both_failure: int = 0  # Failed on both scaffolds
    source_only: int = 0   # Resolved only on source (transfer loss)
    target_only: int = 0   # Resolved only on target (transfer gain)
    
    # Test statistics
    chi_squared: float = 0.0
    p_value: float = 0.0
    
    @property
    def total_instances(self) -> int:
        return self.both_success + self.both_failure + self.source_only + self.target_only
    
    @property
    def source_resolve_rate(self) -> float:
        total = self.total_instances
        if total == 0:
            return 0.0
        return (self.both_success + self.source_only) / total
    
    @property
    def target_resolve_rate(self) -> float:
        total = self.total_instances
        if total == 0:
            return 0.0
        return (self.both_success + self.target_only) / total
    
    @property
    def is_significant(self) -> bool:
        """Whether the difference is statistically significant (p < 0.05)."""
        return self.p_value < 0.05
    
    @property
    def transfer_direction(self) -> str:
        """Direction of transfer effect."""
        if self.source_only > self.target_only:
            return "degradation"
        elif self.target_only > self.source_only:
            return "improvement"
        else:
            return "neutral"
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'both_success': self.both_success,
            'both_failure': self.both_failure,
            'source_only': self.source_only,
            'target_only': self.target_only,
            'total_instances': self.total_instances,
            'source_resolve_rate': self.source_resolve_rate,
            'target_resolve_rate': self.target_resolve_rate,
            'chi_squared': self.chi_squared,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'transfer_direction': self.transfer_direction,
        }


@dataclass
class ErrorModeAnalysis:
    """Analysis of error modes in a run."""
    
    run_name: str
    scaffold: str
    
    # Error counts by category
    error_counts: dict[str, int] = field(default_factory=dict)
    
    # Tool-specific error rates
    tool_error_rates: dict[str, float] = field(default_factory=dict)
    
    # Most common error-causing tools
    error_prone_tools: list[tuple[str, int]] = field(default_factory=list)
    
    # Sample error messages by category
    error_samples: dict[str, list[str]] = field(default_factory=dict)
    
    # Detailed error pattern counts (e.g., "file_not_found", "syntax_error")
    error_pattern_counts: dict[str, int] = field(default_factory=dict)
    
    # Error patterns by tool
    tool_error_patterns: dict[str, dict[str, int]] = field(default_factory=dict)
    
    @property
    def total_errors(self) -> int:
        return sum(self.error_counts.values())
    
    @property
    def top_error_patterns(self) -> list[tuple[str, int]]:
        """Get top error patterns sorted by frequency."""
        return sorted(self.error_pattern_counts.items(), key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'run_name': self.run_name,
            'scaffold': self.scaffold,
            'error_counts': self.error_counts,
            'total_errors': self.total_errors,
            'tool_error_rates': self.tool_error_rates,
            'error_prone_tools': self.error_prone_tools,
            'error_samples': self.error_samples,
            'error_pattern_counts': self.error_pattern_counts,
            'tool_error_patterns': self.tool_error_patterns,
        }


@dataclass
class ToolVocabularyAnalysis:
    """Analysis of tool vocabulary alignment between scaffolds."""
    
    source_scaffold: str
    target_scaffold: str
    
    # Tool vocabularies
    source_tools: set[str] = field(default_factory=set)
    target_tools: set[str] = field(default_factory=set)
    
    # Overlap analysis
    shared_tools: set[str] = field(default_factory=set)
    source_only_tools: set[str] = field(default_factory=set)
    target_only_tools: set[str] = field(default_factory=set)
    
    # Tool usage counts
    source_tool_counts: dict[str, int] = field(default_factory=dict)
    target_tool_counts: dict[str, int] = field(default_factory=dict)
    
    # "Hallucinated" tools: tools called on target that don't exist there
    # (model trying to use source scaffold's tools)
    hallucinated_tools: dict[str, int] = field(default_factory=dict)
    
    # Potential tool mappings (source tool -> likely target equivalent)
    tool_mappings: dict[str, str] = field(default_factory=dict)
    
    @property
    def jaccard_index(self) -> float:
        """Jaccard similarity of tool vocabularies."""
        union = self.source_tools | self.target_tools
        if not union:
            return 0.0
        return len(self.shared_tools) / len(union)
    
    @property
    def hallucination_rate(self) -> float:
        """Rate of hallucinated tool calls on target scaffold."""
        total_hallucinated = sum(self.hallucinated_tools.values())
        total_target_calls = sum(self.target_tool_counts.values())
        if total_target_calls == 0:
            return 0.0
        return total_hallucinated / total_target_calls
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'source_scaffold': self.source_scaffold,
            'target_scaffold': self.target_scaffold,
            'source_tools': list(self.source_tools),
            'target_tools': list(self.target_tools),
            'shared_tools': list(self.shared_tools),
            'source_only_tools': list(self.source_only_tools),
            'target_only_tools': list(self.target_only_tools),
            'jaccard_index': self.jaccard_index,
            'hallucinated_tools': self.hallucinated_tools,
            'hallucination_rate': self.hallucination_rate,
            'tool_mappings': self.tool_mappings,
        }


@dataclass 
class DeepTransferAnalysis:
    """Comprehensive transfer learning analysis."""
    
    source_run_name: str
    target_run_name: str
    base_model: str
    source_scaffold: str
    target_scaffold: str
    
    # McNemar's test results
    mcnemar: McNemarResult = field(default_factory=McNemarResult)
    
    # Per-instance results
    instance_results: list[InstanceTransferResult] = field(default_factory=list)
    
    # Error mode comparison
    source_errors: ErrorModeAnalysis | None = None
    target_errors: ErrorModeAnalysis | None = None
    
    # Tool vocabulary analysis
    vocabulary: ToolVocabularyAnalysis | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            'source_run_name': self.source_run_name,
            'target_run_name': self.target_run_name,
            'base_model': self.base_model,
            'source_scaffold': self.source_scaffold,
            'target_scaffold': self.target_scaffold,
            'mcnemar': self.mcnemar.to_dict(),
            'instance_results_summary': {
                'both_success': self.mcnemar.both_success,
                'both_failure': self.mcnemar.both_failure,
                'transfer_loss': self.mcnemar.source_only,
                'transfer_gain': self.mcnemar.target_only,
            },
            'source_errors': self.source_errors.to_dict() if self.source_errors else None,
            'target_errors': self.target_errors.to_dict() if self.target_errors else None,
            'vocabulary': self.vocabulary.to_dict() if self.vocabulary else None,
        }


class TransferAnalyzer:
    """Perform deep transfer learning analysis."""
    
    # Known tool mappings between scaffolds
    TOOL_MAPPINGS = {
        # R2E-Gym -> nano-agent potential mappings
        ('r2e-gym', 'nano-agent'): {
            'file_editor': 'apply_patch',
            'execute_bash': 'shell',
            'search': 'shell',  # grep/rg via shell
        },
        # nano-agent -> R2E-Gym potential mappings  
        ('nano-agent', 'r2e-gym'): {
            'shell': 'execute_bash',
            'apply_patch': 'file_editor',
        },
    }
    
    # Known valid tools per scaffold
    SCAFFOLD_TOOLS = {
        'nano-agent': {'shell', 'apply_patch'},
        'r2e-gym': {'file_editor', 'execute_bash', 'search', 'finish', 'str_replace'},
        'swe-agent': {'edit', 'view', 'create', 'submit', 'search_file', 'search_dir', 'find_file'},
    }
    
    def analyze_transfer(
        self,
        source_run: Run,
        target_run: Run,
    ) -> DeepTransferAnalysis:
        """Perform comprehensive transfer analysis.
        
        Args:
            source_run: Run on the source (training) scaffold
            target_run: Run on the target scaffold
            
        Returns:
            DeepTransferAnalysis with all metrics
        """
        analysis = DeepTransferAnalysis(
            source_run_name=source_run.name,
            target_run_name=target_run.name,
            base_model=source_run.base_model,
            source_scaffold=source_run.scaffold,
            target_scaffold=target_run.scaffold,
        )
        
        # 1. Instance-level correlation with McNemar's test
        analysis.mcnemar, analysis.instance_results = self._compute_mcnemar(
            source_run, target_run
        )
        
        # 2. Error mode analysis
        analysis.source_errors = self._analyze_errors(source_run)
        analysis.target_errors = self._analyze_errors(target_run)
        
        # 3. Tool vocabulary alignment
        analysis.vocabulary = self._analyze_vocabulary(source_run, target_run)
        
        return analysis
    
    def _compute_mcnemar(
        self,
        source_run: Run,
        target_run: Run,
    ) -> tuple[McNemarResult, list[InstanceTransferResult]]:
        """Compute McNemar's test for paired binary outcomes.
        
        McNemar's test compares paired proportions. The null hypothesis is that
        the probability of success is the same on both scaffolds.
        
        Uses the chi-squared approximation: χ² = (b - c)² / (b + c)
        where b = source_only, c = target_only (discordant pairs)
        """
        # Build lookup for target run
        target_by_id = {t.instance_id: t for t in target_run.trajectories}
        
        result = McNemarResult()
        instance_results = []
        
        for source_traj in source_run.trajectories:
            target_traj = target_by_id.get(source_traj.instance_id)
            if target_traj is None:
                continue  # Instance not in both runs
            
            source_resolved = source_traj.resolved or False
            target_resolved = target_traj.resolved or False
            
            instance_results.append(InstanceTransferResult(
                instance_id=source_traj.instance_id,
                source_resolved=source_resolved,
                target_resolved=target_resolved,
            ))
            
            if source_resolved and target_resolved:
                result.both_success += 1
            elif source_resolved and not target_resolved:
                result.source_only += 1
            elif not source_resolved and target_resolved:
                result.target_only += 1
            else:
                result.both_failure += 1
        
        # Compute McNemar's chi-squared statistic
        b = result.source_only
        c = result.target_only
        
        if b + c > 0:
            # Standard McNemar's test (with continuity correction for small samples)
            result.chi_squared = (abs(b - c) - 1) ** 2 / (b + c) if b + c >= 25 else (b - c) ** 2 / (b + c)
            
            # Compute p-value from chi-squared distribution (1 df)
            result.p_value = self._chi2_p_value(result.chi_squared, df=1)
        else:
            result.chi_squared = 0.0
            result.p_value = 1.0
        
        return result, instance_results
    
    def _chi2_p_value(self, chi2: float, df: int = 1) -> float:
        """Compute p-value from chi-squared statistic.
        
        Uses the regularized incomplete gamma function approximation.
        For df=1, this simplifies to 1 - erf(sqrt(chi2/2)).
        """
        if chi2 <= 0:
            return 1.0
        
        # For df=1, use complementary error function approximation
        x = math.sqrt(chi2 / 2)
        
        # Approximation of erfc(x) = 1 - erf(x)
        # Using Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.3275911 * x)
        coeffs = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
        poly = sum(c * t ** (i + 1) for i, c in enumerate(coeffs))
        erfc = poly * math.exp(-x * x)
        
        return erfc
    
    def _analyze_errors(self, run: Run) -> ErrorModeAnalysis:
        """Analyze error modes in a run."""
        analysis = ErrorModeAnalysis(
            run_name=run.name,
            scaffold=run.scaffold,
        )
        
        error_counts: dict[str, int] = defaultdict(int)
        tool_errors: dict[str, int] = defaultdict(int)
        tool_totals: dict[str, int] = defaultdict(int)
        error_samples: dict[str, list[str]] = defaultdict(list)
        error_pattern_counts: dict[str, int] = defaultdict(int)
        tool_error_patterns: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        valid_tools = self.SCAFFOLD_TOOLS.get(run.scaffold, set())
        
        for traj in run.trajectories:
            # Check trajectory-level errors
            if not traj.has_patch:
                error_counts[ErrorCategory.NO_PATCH_GENERATED.value] += 1
            elif traj.resolved is False:
                error_counts[ErrorCategory.PATCH_FAILED.value] += 1
            
            if traj.exit_reason == 'max_steps':
                error_counts[ErrorCategory.MAX_STEPS_REACHED.value] += 1
            
            # Check tool-level errors
            for tc in traj.get_tool_calls():
                tool_totals[tc.name] += 1
                
                # Check for hallucinated tools
                if valid_tools and tc.name not in valid_tools:
                    error_counts[ErrorCategory.TOOL_NOT_FOUND.value] += 1
                    tool_errors[tc.name] += 1
                    if len(error_samples[ErrorCategory.TOOL_NOT_FOUND.value]) < 5:
                        error_samples[ErrorCategory.TOOL_NOT_FOUND.value].append(
                            f"Called '{tc.name}' (not in {run.scaffold})"
                        )
                
                # Check for execution errors
                elif not tc.success:
                    error_counts[ErrorCategory.TOOL_EXECUTION_ERROR.value] += 1
                    tool_errors[tc.name] += 1
                    
                    # Extract error pattern from result (use raw mode for more detail)
                    error_pattern = extract_error_pattern(tc.result or "", mode="raw")
                    error_pattern_counts[error_pattern] += 1
                    tool_error_patterns[tc.name][error_pattern] += 1
                    
                    if len(error_samples[ErrorCategory.TOOL_EXECUTION_ERROR.value]) < 5:
                        result_preview = (tc.result or "")[:100]
                        error_samples[ErrorCategory.TOOL_EXECUTION_ERROR.value].append(
                            f"{tc.name}: {result_preview}..."
                        )
        
        analysis.error_counts = dict(error_counts)
        analysis.error_samples = dict(error_samples)
        analysis.error_pattern_counts = dict(error_pattern_counts)
        analysis.tool_error_patterns = {
            tool: dict(patterns) for tool, patterns in tool_error_patterns.items()
        }
        
        # Compute tool error rates
        analysis.tool_error_rates = {
            tool: tool_errors.get(tool, 0) / total
            for tool, total in tool_totals.items()
            if total > 0
        }
        
        # Find most error-prone tools
        analysis.error_prone_tools = sorted(
            [(tool, count) for tool, count in tool_errors.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return analysis
    
    def _analyze_vocabulary(
        self,
        source_run: Run,
        target_run: Run,
    ) -> ToolVocabularyAnalysis:
        """Analyze tool vocabulary alignment."""
        analysis = ToolVocabularyAnalysis(
            source_scaffold=source_run.scaffold,
            target_scaffold=target_run.scaffold,
        )
        
        # Collect tool usage from source
        source_counts: dict[str, int] = defaultdict(int)
        for traj in source_run.trajectories:
            for tc in traj.get_tool_calls():
                source_counts[tc.name] += 1
        
        # Collect tool usage from target
        target_counts: dict[str, int] = defaultdict(int)
        for traj in target_run.trajectories:
            for tc in traj.get_tool_calls():
                target_counts[tc.name] += 1
        
        analysis.source_tools = set(source_counts.keys())
        analysis.target_tools = set(target_counts.keys())
        analysis.source_tool_counts = dict(source_counts)
        analysis.target_tool_counts = dict(target_counts)
        
        # Compute overlaps
        analysis.shared_tools = analysis.source_tools & analysis.target_tools
        analysis.source_only_tools = analysis.source_tools - analysis.target_tools
        analysis.target_only_tools = analysis.target_tools - analysis.source_tools
        
        # Check for hallucinated tools on target
        # (tools that exist on source but not on target scaffold's valid set)
        valid_target_tools = self.SCAFFOLD_TOOLS.get(target_run.scaffold, set())
        if valid_target_tools:
            for tool, count in target_counts.items():
                if tool not in valid_target_tools:
                    analysis.hallucinated_tools[tool] = count
        
        # Set known tool mappings
        mapping_key = (source_run.scaffold, target_run.scaffold)
        analysis.tool_mappings = self.TOOL_MAPPINGS.get(mapping_key, {})
        
        return analysis
    
    def get_transfer_loss_instances(
        self,
        analysis: DeepTransferAnalysis,
    ) -> list[str]:
        """Get instance IDs that succeeded on source but failed on target."""
        return [
            r.instance_id for r in analysis.instance_results
            if r.transfer_category == "transfer_loss"
        ]
    
    def get_transfer_gain_instances(
        self,
        analysis: DeepTransferAnalysis,
    ) -> list[str]:
        """Get instance IDs that failed on source but succeeded on target."""
        return [
            r.instance_id for r in analysis.instance_results
            if r.transfer_category == "transfer_gain"
        ]

