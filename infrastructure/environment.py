"""
Environment Auto-Detection for Paragon.

Principle: Never ask what we can detect programmatically.
Reference: CLAUDE.md Section 3

This module automatically detects system capabilities to enable:
- Resource planning (RAM, disk space)
- GPU availability detection
- Network connectivity checks
- Git repository detection
"""
import platform
import sys
import shutil
import socket
import os
from pathlib import Path
from typing import Optional
import msgspec

# Optional dependencies with graceful fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class EnvironmentReport(msgspec.Struct, frozen=True, kw_only=True):
    """Complete environment detection report."""
    os_name: str
    python_version: str
    ram_gb: float
    gpu_available: bool
    gpu_name: Optional[str]
    disk_free_gb: float
    network_available: bool
    git_repo_present: bool
    working_directory: str

    def __str__(self) -> str:
        """Human-readable report format."""
        lines = [
            "=== Paragon Environment Report ===",
            f"OS: {self.os_name}",
            f"Python: {self.python_version}",
            f"RAM: {self.ram_gb:.1f} GB",
            f"GPU: {self.gpu_name if self.gpu_available else 'None (CPU-only mode)'}",
            f"Disk Free: {self.disk_free_gb:.1f} GB",
            f"Network: {'Available' if self.network_available else 'Offline'}",
            f"Git Repo: {'Yes' if self.git_repo_present else 'No'}",
            f"Working Dir: {self.working_directory}",
        ]
        return "\n".join(lines)


class EnvironmentDetector:
    """Auto-detect system environment for resource planning."""

    # Constants for fallback values
    DEFAULT_RAM_GB = 8.0
    DEFAULT_DISK_GB = 10.0
    NETWORK_TEST_HOST = "pypi.org"
    NETWORK_TEST_PORT = 443
    NETWORK_TIMEOUT = 3.0

    def __init__(self, working_dir: Optional[str | Path] = None):
        """
        Initialize detector with optional working directory.

        Args:
            working_dir: Directory to check for git repo. Defaults to cwd.
        """
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()

    def detect(self) -> EnvironmentReport:
        """
        Run all detection methods and return complete report.

        Returns:
            EnvironmentReport with all detected system capabilities.
        """
        gpu_available, gpu_name = self._detect_gpu()

        return EnvironmentReport(
            os_name=self._detect_os(),
            python_version=self._detect_python(),
            ram_gb=self._detect_ram(),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            disk_free_gb=self._detect_disk(),
            network_available=self._detect_network(),
            git_repo_present=self._detect_git(),
            working_directory=str(self.working_dir.resolve()),
        )

    def _detect_os(self) -> str:
        """
        Detect operating system.

        Returns:
            OS name (e.g., 'Linux', 'Darwin', 'Windows').
        """
        return platform.system()

    def _detect_python(self) -> str:
        """
        Detect Python version.

        Returns:
            Python version string (e.g., '3.11.5').
        """
        version_info = sys.version_info
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    def _detect_ram(self) -> float:
        """
        Detect available system RAM in GB.

        Returns:
            Total RAM in GB. Falls back to DEFAULT_RAM_GB if psutil unavailable.
        """
        if not PSUTIL_AVAILABLE:
            # Fallback when psutil not available
            return self.DEFAULT_RAM_GB

        try:
            memory = psutil.virtual_memory()
            return memory.total / (1024 ** 3)  # Convert bytes to GB
        except Exception:
            return self.DEFAULT_RAM_GB

    def _detect_gpu(self) -> tuple[bool, Optional[str]]:
        """
        Detect GPU availability and name.

        Tries multiple detection methods:
        1. PyTorch CUDA detection
        2. nvidia-smi command

        Returns:
            Tuple of (gpu_available: bool, gpu_name: str | None)
        """
        # Try PyTorch first
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    return (True, gpu_name)
            except Exception:
                pass

        # Try nvidia-smi as fallback
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2.0
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                return (True, gpu_name)
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # No GPU detected
        return (False, None)

    def _detect_disk(self) -> float:
        """
        Detect free disk space in GB for working directory.

        Returns:
            Free disk space in GB. Falls back to DEFAULT_DISK_GB on error.
        """
        try:
            usage = shutil.disk_usage(self.working_dir)
            return usage.free / (1024 ** 3)  # Convert bytes to GB
        except Exception:
            return self.DEFAULT_DISK_GB

    def _detect_network(self) -> bool:
        """
        Detect network connectivity via socket test to PyPI.

        Returns:
            True if network available, False otherwise.
        """
        try:
            # Create socket and attempt connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.NETWORK_TIMEOUT)

            # Try to connect to PyPI
            sock.connect((self.NETWORK_TEST_HOST, self.NETWORK_TEST_PORT))
            sock.close()
            return True

        except (socket.timeout, socket.error, OSError):
            return False

    def _detect_git(self) -> bool:
        """
        Detect if working directory is a git repository.

        Returns:
            True if .git directory exists in working_dir or parents.
        """
        # Check current directory and all parents
        current = self.working_dir.resolve()

        while True:
            git_dir = current / ".git"
            if git_dir.exists():
                return True

            # Move to parent directory
            parent = current.parent
            if parent == current:
                # Reached filesystem root
                break
            current = parent

        return False


def detect_environment(working_dir: Optional[str | Path] = None) -> EnvironmentReport:
    """
    Convenience function to detect environment in one call.

    Args:
        working_dir: Optional working directory path. Defaults to cwd.

    Returns:
        EnvironmentReport with all detected capabilities.

    Example:
        >>> report = detect_environment()
        >>> print(report)
        === Paragon Environment Report ===
        OS: Darwin
        Python: 3.11.5
        RAM: 16.0 GB
        GPU: None (CPU-only mode)
        ...
    """
    detector = EnvironmentDetector(working_dir)
    return detector.detect()


# Export main interfaces
__all__ = [
    "EnvironmentReport",
    "EnvironmentDetector",
    "detect_environment",
]
