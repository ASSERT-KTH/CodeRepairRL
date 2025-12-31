"""
Test for SWE-Bench Verified environment setup with Apptainer backend.

This test verifies that the environment setup in CodeRepairRL matches R2E-Gym's setup:
- PATH is set correctly to DOCKER_PATH
- chardet is installed
- ripgrep is installed
- run_tests.sh is executable
- venv symlink is created
- Basic file operations work
- Commands can be executed
"""

import pytest
import os
import sys
import shutil
import logging
from datasets import load_dataset

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nano.env import ApptainerEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected PATH from R2E-Gym
EXPECTED_DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def get_swebench_sample():
    """Load a single sample from SWE-Bench Verified."""
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", streaming=True)
        # Get the first example
        return next(iter(ds))
    except Exception as e:
        pytest.skip(f"Failed to load SWE-Bench Verified dataset: {e}")


@pytest.fixture(scope="module")
def swebench_sample():
    return get_swebench_sample()


@pytest.fixture(scope="module")
def apptainer_env(swebench_sample):
    """Create and setup an Apptainer environment for testing."""
    # Check if apptainer is available
    if not shutil.which("apptainer"):
        pytest.skip("Apptainer not installed")

    instance_id = swebench_sample.get("instance_id", "")
    if not instance_id:
        pytest.skip("Sample does not have instance_id")

    # Construct image name
    image_name = f"docker.io/slimshetty/swebench-verified:sweb.eval.x86_64.{instance_id}"
    workdir = "/testbed"

    logger.info(f"Creating ApptainerEnvironment for {instance_id}")
    logger.info(f"Image: {image_name}")

    # Create environment
    env = ApptainerEnvironment(image=f"docker://{image_name}", workdir=workdir)

    try:
        # Start the environment (this also runs setup)
        env.start()

        # Import and run the setup function
        from src.agents.nano_agent import setup_swebench_environment
        setup_swebench_environment(env)

        yield env

    finally:
        # Cleanup
        logger.info("Stopping environment...")
        env.stop()


def test_apptainer_available():
    """Test that Apptainer is installed."""
    assert shutil.which("apptainer"), "Apptainer not installed"


def test_environment_starts(apptainer_env):
    """Test that the Apptainer environment starts successfully."""
    assert apptainer_env.started, "Environment failed to start"
    logger.info("✓ Environment started successfully")


def test_path_is_set(apptainer_env):
    """Test that PATH is set to DOCKER_PATH."""
    result = apptainer_env.run_shell("echo $PATH", timeout=30)
    assert result.returncode == 0, f"Failed to get PATH: {result.stdout}"

    path = result.stdout.strip()
    logger.info(f"PATH: {path}")

    # Check that the path contains key directories
    assert "/root/.venv/bin" in path, "PATH missing /root/.venv/bin"
    assert "/usr/local/bin" in path, "PATH missing /usr/local/bin"
    logger.info("✓ PATH is set correctly")


def test_venv_symlink_exists(apptainer_env):
    """Test that /root/.venv symlink is created."""
    result = apptainer_env.run_shell("test -L /root/.venv && echo 'EXISTS'", timeout=30)
    assert result.returncode == 0, f"Symlink check failed: {result.stdout}"
    assert "EXISTS" in result.stdout, "/root/.venv symlink does not exist"

    # Verify it points to the right location
    result = apptainer_env.run_shell("readlink /root/.venv", timeout=30)
    assert result.returncode == 0, f"Failed to read symlink: {result.stdout}"
    assert "/opt/miniconda3/envs/testbed" in result.stdout, "Symlink points to wrong location"
    logger.info("✓ /root/.venv symlink exists and points correctly")


def test_chardet_installed(apptainer_env):
    """Test that chardet package is installed."""
    result = apptainer_env.run_shell("python -c 'import chardet; print(chardet.__version__)'", timeout=60)
    assert result.returncode == 0, f"chardet not installed: {result.stdout}"
    logger.info(f"✓ chardet is installed (version: {result.stdout.strip()})")


def test_ripgrep_installed(apptainer_env):
    """Test that ripgrep is installed."""
    result = apptainer_env.run_shell("which rg", timeout=30)
    assert result.returncode == 0, f"ripgrep not installed: {result.stdout}"

    result = apptainer_env.run_shell("rg --version", timeout=30)
    assert result.returncode == 0, f"Failed to get ripgrep version: {result.stdout}"
    logger.info(f"✓ ripgrep is installed: {result.stdout.strip().split()[0]}")


def test_run_tests_executable(apptainer_env):
    """Test that /run_tests.sh exists and is executable."""
    result = apptainer_env.run_shell("test -x /run_tests.sh && echo 'EXECUTABLE'", timeout=30)
    assert result.returncode == 0, f"run_tests.sh check failed: {result.stdout}"
    assert "EXECUTABLE" in result.stdout, "/run_tests.sh is not executable"
    logger.info("✓ /run_tests.sh is executable")


def test_python_version(apptainer_env):
    """Test that Python is accessible and get its version."""
    result = apptainer_env.run_shell("python --version", timeout=30)
    assert result.returncode == 0, f"Failed to get Python version: {result.stdout}"
    logger.info(f"✓ Python version: {result.stdout.strip()}")


def test_git_repository(apptainer_env):
    """Test that the workdir is a git repository."""
    result = apptainer_env.run_shell("test -d .git && echo 'IS_GIT_REPO'", timeout=30)
    assert result.returncode == 0, f"Git check failed: {result.stdout}"
    assert "IS_GIT_REPO" in result.stdout, "Workdir is not a git repository"
    logger.info("✓ Workdir is a git repository")


def test_basic_shell_commands(apptainer_env):
    """Test that basic shell commands work."""
    commands = [
        ("pwd", "/testbed"),
        ("whoami", "root"),
        ("echo 'test'", "test"),
    ]

    for cmd, expected in commands:
        result = apptainer_env.run_shell(cmd, timeout=30)
        assert result.returncode == 0, f"Command '{cmd}' failed: {result.stdout}"
        assert expected in result.stdout, f"Expected '{expected}' in output of '{cmd}'"
        logger.info(f"✓ Command '{cmd}' works")


def test_file_operations(apptainer_env):
    """Test that file read/write operations work."""
    import uuid
    test_file = f"test_file_{uuid.uuid4().hex[:8]}.txt"
    test_content = "Hello from CodeRepairRL test!"

    # Write file
    result = apptainer_env.run_shell(f"echo '{test_content}' > {test_file}", timeout=30)
    assert result.returncode == 0, f"Failed to write file: {result.stdout}"

    # Read file
    result = apptainer_env.run_shell(f"cat {test_file}", timeout=30)
    assert result.returncode == 0, f"Failed to read file: {result.stdout}"
    assert test_content in result.stdout, "File content mismatch"

    # Delete file
    result = apptainer_env.run_shell(f"rm {test_file}", timeout=30)
    assert result.returncode == 0, f"Failed to delete file: {result.stdout}"

    logger.info("✓ File operations work correctly")


def test_python_imports(apptainer_env):
    """Test that common Python packages are available."""
    packages = [
        "sys",
        "os",
        "subprocess",
        "pytest",
    ]

    for package in packages:
        result = apptainer_env.run_shell(f"python -c 'import {package}'", timeout=30)
        # Some packages might not be installed, but core ones should be
        if package in ["sys", "os", "subprocess"]:
            assert result.returncode == 0, f"Failed to import {package}: {result.stdout}"
        logger.info(f"✓ Can import {package}" if result.returncode == 0 else f"- {package} not available (OK)")


def test_workdir_location(apptainer_env):
    """Test that we're in the correct workdir."""
    result = apptainer_env.run_shell("pwd", timeout=30)
    assert result.returncode == 0, f"Failed to get pwd: {result.stdout}"
    assert "/testbed" in result.stdout, f"Not in correct workdir. Got: {result.stdout}"
    logger.info("✓ Workdir is /testbed")


def test_repository_files_exist(apptainer_env):
    """Test that repository files exist in the workdir."""
    result = apptainer_env.run_shell("ls -la", timeout=30)
    assert result.returncode == 0, f"Failed to list files: {result.stdout}"

    # Should have some files (not empty directory)
    lines = result.stdout.strip().split("\n")
    assert len(lines) > 3, "Workdir appears to be empty"  # > 3 to account for ., .., and at least one file
    logger.info("✓ Repository files exist in workdir")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
