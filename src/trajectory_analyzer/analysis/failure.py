"""Failure taxonomy analysis: categorize why trajectories fail."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models import Run, Trajectory, Instance
from .dynamics import extract_files_from_patch, extract_files_edited

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """High-level failure categories."""
    
    # Localization failures
    WRONG_FILES = "wrong_files"           # Edited wrong files entirely
    MISSED_FILES = "missed_files"         # Didn't edit all necessary files
    NO_LOCALIZATION = "no_localization"   # Never found any relevant files
    
    # Edit failures
    NO_EDIT_ATTEMPTED = "no_edit_attempted"     # Never tried to edit
    EDIT_SYNTAX_ERROR = "edit_syntax_error"     # Edit failed due to syntax
    EDIT_NOT_APPLIED = "edit_not_applied"       # Edit command failed
    INCOMPLETE_EDIT = "incomplete_edit"         # Partial fix (some files correct)
    WRONG_EDIT = "wrong_edit"                   # Edited right files, wrong content
    
    # Process failures
    MAX_STEPS_REACHED = "max_steps_reached"     # Hit step limit
    MAX_TOKENS_REACHED = "max_tokens_reached"   # Hit token limit
    STUCK_IN_LOOP = "stuck_in_loop"             # Repeated same actions
    GAVE_UP_EARLY = "gave_up_early"             # Submitted without trying much
    
    # Error-related failures
    UNRECOVERED_ERROR = "unrecovered_error"     # Fatal error not recovered from
    CASCADE_ERRORS = "cascade_errors"           # Multiple consecutive errors
    
    # Patch quality failures
    PATCH_TOO_LARGE = "patch_too_large"         # Over-patched
    PATCH_BREAKS_TESTS = "patch_breaks_tests"   # Patch applied but tests fail
    
    # Unknown
    UNKNOWN = "unknown"


@dataclass
class FailureDiagnosis:
    """Diagnosis of why a single trajectory failed."""
    
    instance_id: str
    resolved: bool
    
    # Primary failure category
    primary_category: FailureCategory = FailureCategory.UNKNOWN
    
    # Secondary categories (can have multiple contributing factors)
    secondary_categories: list[FailureCategory] = field(default_factory=list)
    
    # Detailed metrics
    oracle_files: set[str] = field(default_factory=set)
    edited_files: set[str] = field(default_factory=set)
    
    # Localization metrics
    files_overlap: int = 0           # |edited ∩ oracle|
    files_missed: int = 0            # |oracle - edited|
    files_extra: int = 0             # |edited - oracle|
    
    # Edit metrics
    edit_attempts: int = 0
    successful_edits: int = 0
    failed_edits: int = 0
    
    # Error metrics
    total_errors: int = 0
    max_consecutive_errors: int = 0
    final_step_had_error: bool = False
    
    # Process metrics
    total_steps: int = 0
    total_tool_calls: int = 0
    unique_tool_calls: int = 0       # Distinct (tool, args) pairs
    
    # Loop detection
    repeated_actions: int = 0        # Same tool call repeated
    potential_loop: bool = False
    
    # Patch metrics
    has_patch: bool = False
    patch_line_count: int = 0
    oracle_line_count: int = 0
    
    # Evidence/explanation
    evidence: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'instance_id': self.instance_id,
            'resolved': self.resolved,
            'primary_category': self.primary_category.value,
            'secondary_categories': [c.value for c in self.secondary_categories],
            'files_overlap': self.files_overlap,
            'files_missed': self.files_missed,
            'files_extra': self.files_extra,
            'edit_attempts': self.edit_attempts,
            'successful_edits': self.successful_edits,
            'failed_edits': self.failed_edits,
            'total_errors': self.total_errors,
            'max_consecutive_errors': self.max_consecutive_errors,
            'total_steps': self.total_steps,
            'repeated_actions': self.repeated_actions,
            'potential_loop': self.potential_loop,
            'has_patch': self.has_patch,
            'evidence': self.evidence,
        }


@dataclass
class FailureTaxonomy:
    """Aggregated failure analysis for a run."""
    
    run_name: str
    scaffold: str
    base_model: str
    
    # Counts by category
    category_counts: dict[str, int] = field(default_factory=dict)
    
    # Resolved vs unresolved
    total_trajectories: int = 0
    resolved_count: int = 0
    unresolved_count: int = 0
    
    # Per-trajectory diagnoses
    diagnoses: list[FailureDiagnosis] = field(default_factory=list)
    
    # Aggregated insights
    most_common_primary: list[tuple[str, int]] = field(default_factory=list)
    most_common_secondary: list[tuple[str, int]] = field(default_factory=list)
    
    # Detailed breakdowns
    localization_failure_rate: float = 0.0   # % failures due to localization
    edit_failure_rate: float = 0.0           # % failures due to edit issues
    process_failure_rate: float = 0.0        # % failures due to process issues
    
    # Recovery analysis
    errors_leading_to_failure: int = 0
    loops_leading_to_failure: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_name': self.run_name,
            'scaffold': self.scaffold,
            'base_model': self.base_model,
            'total_trajectories': self.total_trajectories,
            'resolved_count': self.resolved_count,
            'unresolved_count': self.unresolved_count,
            'category_counts': self.category_counts,
            'most_common_primary': self.most_common_primary,
            'most_common_secondary': self.most_common_secondary,
            'localization_failure_rate': self.localization_failure_rate,
            'edit_failure_rate': self.edit_failure_rate,
            'process_failure_rate': self.process_failure_rate,
        }


# Edit tool names across different scaffolds
EDIT_TOOLS = {
    'str_replace_editor', 'apply_patch', 'file_editor', 'edit', 'create',
    'str_replace', 'insert', 'write_file',
}


class FailureTaxonomyAnalyzer:
    """Analyze and categorize trajectory failures."""
    
    def analyze_run(
        self,
        run: Run,
        instances: dict[str, Instance] | None = None,
    ) -> FailureTaxonomy:
        """Perform failure taxonomy analysis on a run.
        
        Args:
            run: Run to analyze
            instances: Optional instances dict for ground truth
            
        Returns:
            FailureTaxonomy with categorized failures
        """
        taxonomy = FailureTaxonomy(
            run_name=run.name,
            scaffold=run.scaffold,
            base_model=run.base_model,
            total_trajectories=len(run.trajectories),
        )
        
        category_counter = Counter()
        secondary_counter = Counter()
        
        for traj in run.trajectories:
            instance = instances.get(traj.instance_id) if instances else None
            
            diagnosis = self._diagnose_trajectory(traj, instance)
            taxonomy.diagnoses.append(diagnosis)
            
            if traj.resolved:
                taxonomy.resolved_count += 1
            else:
                taxonomy.unresolved_count += 1
                category_counter[diagnosis.primary_category.value] += 1
                for cat in diagnosis.secondary_categories:
                    secondary_counter[cat.value] += 1
        
        # Aggregate results
        taxonomy.category_counts = dict(category_counter)
        taxonomy.most_common_primary = category_counter.most_common()
        taxonomy.most_common_secondary = secondary_counter.most_common()
        
        # Compute rates
        if taxonomy.unresolved_count > 0:
            localization_cats = {
                FailureCategory.WRONG_FILES.value,
                FailureCategory.MISSED_FILES.value,
                FailureCategory.NO_LOCALIZATION.value,
            }
            edit_cats = {
                FailureCategory.NO_EDIT_ATTEMPTED.value,
                FailureCategory.EDIT_SYNTAX_ERROR.value,
                FailureCategory.EDIT_NOT_APPLIED.value,
                FailureCategory.INCOMPLETE_EDIT.value,
                FailureCategory.WRONG_EDIT.value,
            }
            process_cats = {
                FailureCategory.MAX_STEPS_REACHED.value,
                FailureCategory.MAX_TOKENS_REACHED.value,
                FailureCategory.STUCK_IN_LOOP.value,
                FailureCategory.GAVE_UP_EARLY.value,
            }
            
            loc_count = sum(taxonomy.category_counts.get(c, 0) for c in localization_cats)
            edit_count = sum(taxonomy.category_counts.get(c, 0) for c in edit_cats)
            process_count = sum(taxonomy.category_counts.get(c, 0) for c in process_cats)
            
            taxonomy.localization_failure_rate = loc_count / taxonomy.unresolved_count
            taxonomy.edit_failure_rate = edit_count / taxonomy.unresolved_count
            taxonomy.process_failure_rate = process_count / taxonomy.unresolved_count
            
            # Count specific issues
            taxonomy.loops_leading_to_failure = sum(
                1 for d in taxonomy.diagnoses 
                if not d.resolved and d.potential_loop
            )
            taxonomy.errors_leading_to_failure = sum(
                1 for d in taxonomy.diagnoses
                if not d.resolved and d.max_consecutive_errors >= 3
            )
        
        return taxonomy
    
    def _diagnose_trajectory(
        self,
        traj: Trajectory,
        instance: Instance | None,
    ) -> FailureDiagnosis:
        """Diagnose a single trajectory's failure mode.
        
        Args:
            traj: Trajectory to diagnose
            instance: Optional instance with ground truth
            
        Returns:
            FailureDiagnosis with categorization
        """
        diagnosis = FailureDiagnosis(
            instance_id=traj.instance_id,
            resolved=traj.resolved or False,
            total_steps=len(traj.steps),
            has_patch=traj.has_patch,
        )
        
        # Get oracle files if available
        if instance and instance.patch:
            diagnosis.oracle_files = extract_files_from_patch(instance.patch)
            diagnosis.oracle_line_count = len(instance.patch.split('\n'))
        
        # Get edited files
        diagnosis.edited_files = extract_files_edited(traj)
        
        # Compute file overlap
        if diagnosis.oracle_files:
            overlap = diagnosis.oracle_files & diagnosis.edited_files
            diagnosis.files_overlap = len(overlap)
            diagnosis.files_missed = len(diagnosis.oracle_files - diagnosis.edited_files)
            diagnosis.files_extra = len(diagnosis.edited_files - diagnosis.oracle_files)
        
        # Analyze tool calls
        tool_call_signatures = Counter()
        consecutive_errors = 0
        max_consecutive = 0
        
        for step in traj.steps:
            for tc in step.tool_calls:
                diagnosis.total_tool_calls += 1
                
                # Track edit attempts
                if tc.name in EDIT_TOOLS:
                    diagnosis.edit_attempts += 1
                    if tc.success:
                        diagnosis.successful_edits += 1
                    else:
                        diagnosis.failed_edits += 1
                
                # Track errors
                if not tc.success:
                    diagnosis.total_errors += 1
                    consecutive_errors += 1
                    max_consecutive = max(max_consecutive, consecutive_errors)
                else:
                    consecutive_errors = 0
                
                # Track repeated actions (loop detection)
                sig = (tc.name, str(sorted(tc.arguments.items())))
                tool_call_signatures[sig] += 1
        
        diagnosis.max_consecutive_errors = max_consecutive
        diagnosis.unique_tool_calls = len(tool_call_signatures)
        
        # Detect repeated actions
        diagnosis.repeated_actions = sum(
            count - 1 for count in tool_call_signatures.values() if count > 1
        )
        diagnosis.potential_loop = diagnosis.repeated_actions > 5
        
        # Check if final step had error
        if traj.steps:
            last_step = traj.steps[-1]
            diagnosis.final_step_had_error = any(
                not tc.success for tc in last_step.tool_calls
            )
        
        # Get patch size
        if traj.generated_patch:
            diagnosis.patch_line_count = len(traj.generated_patch.split('\n'))
        
        # Determine failure category
        if traj.resolved:
            diagnosis.primary_category = FailureCategory.UNKNOWN  # Not a failure
        else:
            diagnosis.primary_category = self._categorize_failure(diagnosis, traj)
            diagnosis.secondary_categories = self._find_secondary_categories(diagnosis, traj)
        
        return diagnosis
    
    def _categorize_failure(
        self,
        diagnosis: FailureDiagnosis,
        traj: Trajectory,
    ) -> FailureCategory:
        """Determine the primary failure category.
        
        Args:
            diagnosis: Diagnosis with computed metrics
            traj: Original trajectory
            
        Returns:
            Primary FailureCategory
        """
        # Check for process failures first (most definitive)
        if traj.exit_reason == 'max_steps':
            diagnosis.evidence.append("Hit maximum step limit")
            return FailureCategory.MAX_STEPS_REACHED
        
        if traj.exit_reason == 'max_tokens':
            diagnosis.evidence.append("Hit maximum token limit")
            return FailureCategory.MAX_TOKENS_REACHED
        
        # Check for loop
        if diagnosis.potential_loop:
            diagnosis.evidence.append(f"Detected {diagnosis.repeated_actions} repeated actions")
            return FailureCategory.STUCK_IN_LOOP
        
        # Check for no edit attempts
        if diagnosis.edit_attempts == 0:
            diagnosis.evidence.append("No edit attempts made")
            return FailureCategory.NO_EDIT_ATTEMPTED
        
        # Check for edit failures
        if diagnosis.failed_edits > 0 and diagnosis.successful_edits == 0:
            diagnosis.evidence.append(f"All {diagnosis.failed_edits} edit attempts failed")
            return FailureCategory.EDIT_NOT_APPLIED
        
        # Check for localization issues (with oracle)
        if diagnosis.oracle_files:
            if not diagnosis.edited_files:
                diagnosis.evidence.append("No files were edited")
                return FailureCategory.NO_LOCALIZATION
            
            if diagnosis.files_overlap == 0:
                diagnosis.evidence.append(
                    f"Edited wrong files: {diagnosis.edited_files} vs oracle {diagnosis.oracle_files}"
                )
                return FailureCategory.WRONG_FILES
            
            if diagnosis.files_missed > 0:
                diagnosis.evidence.append(
                    f"Missed {diagnosis.files_missed} files that needed editing"
                )
                return FailureCategory.MISSED_FILES
            
            if diagnosis.files_overlap == len(diagnosis.oracle_files):
                # Edited right files but wrong content
                diagnosis.evidence.append("Edited correct files but wrong content")
                return FailureCategory.WRONG_EDIT
        
        # Check for error-related failures
        if diagnosis.max_consecutive_errors >= 3:
            diagnosis.evidence.append(
                f"Had {diagnosis.max_consecutive_errors} consecutive errors"
            )
            return FailureCategory.CASCADE_ERRORS
        
        if diagnosis.final_step_had_error:
            diagnosis.evidence.append("Final step ended with error")
            return FailureCategory.UNRECOVERED_ERROR
        
        # Check for early giving up
        if diagnosis.total_steps < 5 and diagnosis.edit_attempts < 2:
            diagnosis.evidence.append(
                f"Gave up after only {diagnosis.total_steps} steps"
            )
            return FailureCategory.GAVE_UP_EARLY
        
        # Check for over-patching
        if diagnosis.oracle_line_count > 0 and diagnosis.patch_line_count > 0:
            if diagnosis.patch_line_count > diagnosis.oracle_line_count * 3:
                diagnosis.evidence.append(
                    f"Patch too large: {diagnosis.patch_line_count} lines vs oracle {diagnosis.oracle_line_count}"
                )
                return FailureCategory.PATCH_TOO_LARGE
        
        # Default: we have a patch but tests fail
        if diagnosis.has_patch:
            diagnosis.evidence.append("Patch was generated but tests failed")
            return FailureCategory.PATCH_BREAKS_TESTS
        
        return FailureCategory.UNKNOWN
    
    def _find_secondary_categories(
        self,
        diagnosis: FailureDiagnosis,
        traj: Trajectory,
    ) -> list[FailureCategory]:
        """Find secondary contributing factors.
        
        Args:
            diagnosis: Diagnosis with computed metrics
            traj: Original trajectory
            
        Returns:
            List of secondary FailureCategory
        """
        secondary = []
        primary = diagnosis.primary_category
        
        # Check each potential secondary cause
        if primary != FailureCategory.STUCK_IN_LOOP and diagnosis.potential_loop:
            secondary.append(FailureCategory.STUCK_IN_LOOP)
        
        if primary != FailureCategory.CASCADE_ERRORS and diagnosis.max_consecutive_errors >= 3:
            secondary.append(FailureCategory.CASCADE_ERRORS)
        
        if primary != FailureCategory.MISSED_FILES and diagnosis.files_missed > 0:
            secondary.append(FailureCategory.MISSED_FILES)
        
        if primary != FailureCategory.EDIT_NOT_APPLIED and diagnosis.failed_edits > diagnosis.successful_edits:
            secondary.append(FailureCategory.EDIT_NOT_APPLIED)
        
        return secondary
    
    def analyze_runs(
        self,
        runs: list[Run],
        instances: dict[str, Instance] | None = None,
    ) -> list[FailureTaxonomy]:
        """Analyze multiple runs.
        
        Args:
            runs: List of runs to analyze
            instances: Optional instances dict
            
        Returns:
            List of FailureTaxonomy objects
        """
        taxonomies = []
        for run in runs:
            logger.info(f"Analyzing failures for run: {run.name}")
            taxonomy = self.analyze_run(run, instances)
            taxonomies.append(taxonomy)
        
        return taxonomies
