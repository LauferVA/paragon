"""
Unit tests for infrastructure.environment module.

Tests environment detection with various conditions:
- Successful detection with all dependencies
- Graceful fallback when dependencies missing
- Edge cases and error handling
"""
import sys
import platform
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from infrastructure.environment import (
    EnvironmentDetector,
    EnvironmentReport,
    detect_environment,
)


class TestEnvironmentDetector:
    """Test suite for EnvironmentDetector class."""

    def test_detect_os(self):
        """Test OS detection returns valid platform name."""
        detector = EnvironmentDetector()
        os_name = detector._detect_os()

        # Should be one of the major platforms
        assert os_name in ["Linux", "Darwin", "Windows", "FreeBSD"]
        # Should match platform.system()
        assert os_name == platform.system()

    def test_detect_python(self):
        """Test Python version detection matches sys.version_info."""
        detector = EnvironmentDetector()
        py_version = detector._detect_python()

        # Parse version string
        major, minor, micro = py_version.split(".")
        assert int(major) == sys.version_info.major
        assert int(minor) == sys.version_info.minor
        assert int(micro) == sys.version_info.micro

        # Check format
        assert py_version.count(".") == 2

    def test_detect_ram_with_psutil(self):
        """Test RAM detection when psutil is available."""
        detector = EnvironmentDetector()
        ram_gb = detector._detect_ram()

        # RAM should be positive and reasonable (0.1 GB to 1024 GB)
        assert 0.1 <= ram_gb <= 1024.0
        assert isinstance(ram_gb, float)

    @patch("infrastructure.environment.PSUTIL_AVAILABLE", False)
    def test_detect_ram_without_psutil(self):
        """Test RAM detection falls back when psutil unavailable."""
        detector = EnvironmentDetector()
        ram_gb = detector._detect_ram()

        # Should use default fallback
        assert ram_gb == EnvironmentDetector.DEFAULT_RAM_GB

    def test_detect_disk(self):
        """Test disk space detection for working directory."""
        detector = EnvironmentDetector()
        disk_gb = detector._detect_disk()

        # Disk space should be positive
        assert disk_gb > 0
        assert isinstance(disk_gb, float)

    def test_detect_disk_with_custom_path(self, tmp_path):
        """Test disk detection with custom working directory."""
        detector = EnvironmentDetector(working_dir=tmp_path)
        disk_gb = detector._detect_disk()

        # Should successfully detect for temp path
        assert disk_gb > 0

    def test_detect_disk_fallback_on_error(self):
        """Test disk detection fallback on invalid path."""
        # Use a path that's unlikely to exist
        detector = EnvironmentDetector(working_dir="/invalid/nonexistent/path/xyz123")

        with patch("shutil.disk_usage", side_effect=Exception("Disk error")):
            disk_gb = detector._detect_disk()

            # Should use default fallback
            assert disk_gb == EnvironmentDetector.DEFAULT_DISK_GB

    def test_detect_gpu_no_gpu(self):
        """Test GPU detection when no GPU present (common case)."""
        detector = EnvironmentDetector()
        gpu_available, gpu_name = detector._detect_gpu()

        # Results should be consistent types
        assert isinstance(gpu_available, bool)
        assert gpu_name is None or isinstance(gpu_name, str)

        # If GPU available, should have a name
        if gpu_available:
            assert gpu_name is not None
            assert len(gpu_name) > 0

    @patch("infrastructure.environment.TORCH_AVAILABLE", False)
    def test_detect_gpu_without_torch(self):
        """Test GPU detection falls back to nvidia-smi when torch unavailable."""
        detector = EnvironmentDetector()
        gpu_available, gpu_name = detector._detect_gpu()

        # Should still work (tries nvidia-smi)
        assert isinstance(gpu_available, bool)
        assert gpu_name is None or isinstance(gpu_name, str)

    def test_detect_network(self):
        """Test network detection via PyPI socket test."""
        detector = EnvironmentDetector()
        network_available = detector._detect_network()

        # Network detection should return boolean
        assert isinstance(network_available, bool)

        # If running in CI or with internet, should be True
        # (but we don't assert True since test might run offline)

    @patch("socket.socket")
    def test_detect_network_offline(self, mock_socket):
        """Test network detection when offline."""
        # Mock socket to raise connection error
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = OSError("Network unreachable")
        mock_socket.return_value = mock_sock

        detector = EnvironmentDetector()
        network_available = detector._detect_network()

        # Should detect as offline
        assert network_available is False

    @patch("socket.socket")
    def test_detect_network_timeout(self, mock_socket):
        """Test network detection handles timeout gracefully."""
        # Mock socket to raise timeout
        import socket
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = socket.timeout("Connection timeout")
        mock_socket.return_value = mock_sock

        detector = EnvironmentDetector()
        network_available = detector._detect_network()

        # Should detect as offline on timeout
        assert network_available is False

    def test_detect_git_in_repo(self):
        """Test git detection in actual git repository."""
        # This test file is in a git repo
        detector = EnvironmentDetector(working_dir=Path(__file__).parent)
        git_present = detector._detect_git()

        # Should detect .git in project root
        assert git_present is True

    def test_detect_git_not_in_repo(self, tmp_path):
        """Test git detection in non-git directory."""
        detector = EnvironmentDetector(working_dir=tmp_path)
        git_present = detector._detect_git()

        # Temp directory is not a git repo
        assert git_present is False

    def test_detect_full_report(self):
        """Test full detection returns complete EnvironmentReport."""
        detector = EnvironmentDetector()
        report = detector.detect()

        # Verify report structure
        assert isinstance(report, EnvironmentReport)

        # Check all fields are populated
        assert report.os_name is not None
        assert report.python_version is not None
        assert report.ram_gb > 0
        assert isinstance(report.gpu_available, bool)
        assert report.disk_free_gb > 0
        assert isinstance(report.network_available, bool)
        assert isinstance(report.git_repo_present, bool)
        assert report.working_directory is not None

    def test_detect_with_custom_working_dir(self, tmp_path):
        """Test detection with custom working directory."""
        detector = EnvironmentDetector(working_dir=tmp_path)
        report = detector.detect()

        # Working directory should be the temp path
        assert report.working_directory == str(tmp_path.resolve())

    def test_detect_convenience_function(self):
        """Test convenience function detect_environment()."""
        report = detect_environment()

        # Should return valid EnvironmentReport
        assert isinstance(report, EnvironmentReport)
        assert report.os_name is not None


class TestEnvironmentReport:
    """Test suite for EnvironmentReport struct."""

    def test_report_is_frozen(self):
        """Test that EnvironmentReport is immutable (frozen)."""
        report = EnvironmentReport(
            os_name="Linux",
            python_version="3.11.0",
            ram_gb=16.0,
            gpu_available=False,
            gpu_name=None,
            disk_free_gb=100.0,
            network_available=True,
            git_repo_present=True,
            working_directory="/tmp",
        )

        # Should raise error when trying to modify
        with pytest.raises((AttributeError, TypeError)):
            report.os_name = "Windows"

    def test_report_str_format(self):
        """Test human-readable string format."""
        report = EnvironmentReport(
            os_name="Darwin",
            python_version="3.11.5",
            ram_gb=16.0,
            gpu_available=False,
            gpu_name=None,
            disk_free_gb=250.5,
            network_available=True,
            git_repo_present=True,
            working_directory="/Users/test/paragon",
        )

        report_str = str(report)

        # Should contain key information
        assert "Darwin" in report_str
        assert "3.11.5" in report_str
        assert "16.0 GB" in report_str
        assert "CPU-only mode" in report_str
        assert "250.5 GB" in report_str
        assert "Available" in report_str
        assert "/Users/test/paragon" in report_str

    def test_report_str_format_with_gpu(self):
        """Test string format when GPU is present."""
        report = EnvironmentReport(
            os_name="Linux",
            python_version="3.11.0",
            ram_gb=32.0,
            gpu_available=True,
            gpu_name="NVIDIA GeForce RTX 4090",
            disk_free_gb=500.0,
            network_available=True,
            git_repo_present=True,
            working_directory="/home/user/paragon",
        )

        report_str = str(report)

        # Should show GPU name
        assert "NVIDIA GeForce RTX 4090" in report_str
        assert "CPU-only mode" not in report_str

    def test_report_msgspec_serialization(self):
        """Test that report can be serialized with msgspec."""
        import msgspec

        report = EnvironmentReport(
            os_name="Linux",
            python_version="3.11.0",
            ram_gb=16.0,
            gpu_available=False,
            gpu_name=None,
            disk_free_gb=100.0,
            network_available=True,
            git_repo_present=True,
            working_directory="/tmp",
        )

        # Encode to JSON
        encoded = msgspec.json.encode(report)
        assert isinstance(encoded, bytes)

        # Decode back
        decoded = msgspec.json.decode(encoded, type=EnvironmentReport)
        assert decoded.os_name == report.os_name
        assert decoded.python_version == report.python_version
        assert decoded.ram_gb == report.ram_gb


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_detector_with_none_working_dir(self):
        """Test detector with None working_dir uses cwd."""
        detector = EnvironmentDetector(working_dir=None)
        assert detector.working_dir == Path.cwd()

    def test_detector_with_string_path(self, tmp_path):
        """Test detector accepts string path."""
        detector = EnvironmentDetector(working_dir=str(tmp_path))
        assert detector.working_dir == tmp_path

    def test_detector_with_path_object(self, tmp_path):
        """Test detector accepts Path object."""
        detector = EnvironmentDetector(working_dir=tmp_path)
        assert detector.working_dir == tmp_path

    @patch("infrastructure.environment.PSUTIL_AVAILABLE", False)
    @patch("infrastructure.environment.TORCH_AVAILABLE", False)
    def test_all_fallbacks_active(self):
        """Test detection works with all optional dependencies missing."""
        detector = EnvironmentDetector()
        report = detector.detect()

        # Should still return valid report with fallback values
        assert isinstance(report, EnvironmentReport)
        assert report.ram_gb == EnvironmentDetector.DEFAULT_RAM_GB
        assert report.gpu_available is False
        assert report.gpu_name is None

        # These should still work (no optional deps)
        assert report.os_name is not None
        assert report.python_version is not None
        assert report.disk_free_gb >= 0
