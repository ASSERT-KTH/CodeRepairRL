"""Instance model representing a SWE-bench problem instance with difficulty metrics."""

import re
from dataclasses import dataclass, field
from typing import Any


def count_files_in_patch(patch: str) -> int:
    """Count the number of distinct files changed in a patch.
    
    Args:
        patch: Unified diff patch string
        
    Returns:
        Number of distinct files changed
    """
    if not patch:
        return 0
    
    # Pattern to match file headers in unified diff format
    # Matches: --- a/path/to/file.py or +++ b/path/to/file.py
    file_pattern = re.compile(r'^(?:---|\+\+\+)\s+[ab]/(.+)$', re.MULTILINE)
    
    files = set()
    for match in file_pattern.finditer(patch):
        file_path = match.group(1).strip()
        # Remove timestamps if present (--- a/file.py\t2024-01-01 12:00:00.000000000 +0000)
        file_path = file_path.split('\t')[0].strip()
        if file_path and file_path != '/dev/null':
            files.add(file_path)
    
    return len(files)


def count_lines_in_patch(patch: str) -> int:
    """Count the total number of lines changed (added + removed) in a patch.
    
    Args:
        patch: Unified diff patch string
        
    Returns:
        Total number of lines changed (additions + deletions)
    """
    if not patch:
        return 0
    
    lines_changed = 0
    
    # Pattern to match hunk headers: @@ -start,count +start,count @@
    # And individual diff lines: + (added), - (removed)
    # Note: We count actual line changes, not context lines
    
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            lines_changed += 1
        elif line.startswith('-') and not line.startswith('---'):
            lines_changed += 1
    
    return lines_changed


def count_additions_in_patch(patch: str) -> int:
    """Count the number of lines added in a patch.
    
    Args:
        patch: Unified diff patch string
        
    Returns:
        Number of lines added
    """
    if not patch:
        return 0
    
    additions = 0
    for line in patch.split('\n'):
        if line.startswith('+') and not line.startswith('+++'):
            additions += 1
    
    return additions


def count_deletions_in_patch(patch: str) -> int:
    """Count the number of lines deleted in a patch.
    
    Args:
        patch: Unified diff patch string
        
    Returns:
        Number of lines deleted
    """
    if not patch:
        return 0
    
    deletions = 0
    for line in patch.split('\n'):
        if line.startswith('-') and not line.startswith('---'):
            deletions += 1
    
    return deletions


@dataclass
class Instance:
    """Represents a SWE-bench problem instance with ground truth and difficulty metrics."""
    
    instance_id: str
    """SWE-bench instance identifier (e.g., 'astropy__astropy-14096')."""
    
    patch: str
    """Ground truth patch (unified diff format)."""
    
    test_patch: str | None = None
    """Test patch (unified diff format) if available."""
    
    # Pre-computed difficulty metrics
    num_files_changed: int = 0
    """Number of distinct files changed in the patch."""
    
    num_lines_changed: int = 0
    """Total number of lines changed (additions + deletions)."""
    
    num_additions: int = 0
    """Number of lines added."""
    
    num_deletions: int = 0
    """Number of lines deleted."""
    
    def __post_init__(self):
        """Compute difficulty metrics from patch if not already set."""
        if self.patch and self.num_files_changed == 0:
            self.num_files_changed = count_files_in_patch(self.patch)
            self.num_lines_changed = count_lines_in_patch(self.patch)
            self.num_additions = count_additions_in_patch(self.patch)
            self.num_deletions = count_deletions_in_patch(self.patch)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'instance_id': self.instance_id,
            'patch': self.patch,
            'test_patch': self.test_patch,
            'num_files_changed': self.num_files_changed,
            'num_lines_changed': self.num_lines_changed,
            'num_additions': self.num_additions,
            'num_deletions': self.num_deletions,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Instance':
        """Create Instance from dictionary."""
        return cls(
            instance_id=data['instance_id'],
            patch=data.get('patch', ''),
            test_patch=data.get('test_patch'),
            num_files_changed=data.get('num_files_changed', 0),
            num_lines_changed=data.get('num_lines_changed', 0),
            num_additions=data.get('num_additions', 0),
            num_deletions=data.get('num_deletions', 0),
        )

