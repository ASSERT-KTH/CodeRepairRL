import git
import shutil
import tempfile
import subprocess
from typing import Optional


def handle_to_url(repo_handle: str) -> str:
    return f"https://github.com/{repo_handle}.git"

def clone_repo_at_commit(repo_url: str, commit_id: str, target_dir: Optional[str] = None) -> str:
    """
    Fast clone method with no git history. Initializes an empty repo, adds the remote then fetches the specific commit and checks it out.
    
    Args:
        repo_url: Repository URL
        commit_id: Commit hash to checkout
        target_dir: Optional target directory. If None, creates a temporary directory.
        
    Returns:
        Path to the cloned repository
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp()
    
    
    repo = git.Repo.init(target_dir)
    origin = repo.create_remote('origin', repo_url)
    origin.fetch(commit_id, depth=1)
    repo.git.checkout(commit_id)

    return target_dir

def clean_repo_dir(repo_path: str):
    """Clean tempfolder"""
    assert repo_path.startswith("/tmp/") or repo_path.startswith("/var/folders/") or repo_path.startswith("/local/tmp") or repo_path.count(".tmp/") > 0, "For safety, repo_path must be a temporary directory"
    try:
        shutil.rmtree(repo_path)
    except:
        pass

def resolve_git_commit_hash(git_commit_hash: Optional[str] = None) -> str:
    """To know in which commit the code was run"""
    if git_commit_hash is not None:
        return git_commit_hash
    
    try: return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except: return "unknown"  

def get_staged_diff(repo_path: str) -> str:
    """
    Get a diff of all staged changes in the repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        String representation of the diff for all staged changes
    """
    repo = git.Repo(repo_path)
    return repo.git.diff('--staged')  

def get_head_commit_diff(repo_path: str) -> str:
    """
    Get the diff for the HEAD commit in the repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        String representation of the changes introduced by the HEAD commit
    """
    repo = git.Repo(repo_path)  
    return repo.git.show('HEAD', "--format=")  
