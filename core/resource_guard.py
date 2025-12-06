"""
PARAGON OPTIMIZER - Resource Guard

System stability monitor that prevents OOM crashes and CPU thrashing.
Runs a background thread monitoring RAM and CPU usage.

Design:
- Background monitoring thread polls system resources
- Emits PAUSE signal when thresholds exceeded for sustained duration
- Blocks orchestrator until resources free up
- Thread-safe signal handling

Architecture:
    Orchestrator Loop
        |
        v
    ResourceGuard.get_signal() -> OK | PAUSE
        |
        v (if PAUSE)
    ResourceGuard.wait_for_resources()
        |
        v (blocks until OK)
    Resume Orchestration

Configuration:
    All thresholds configurable via config/paragon.toml [system.resources]
"""
import psutil
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# RESOURCE SIGNAL
# =============================================================================

class ResourceSignal(Enum):
    """System resource availability signals."""
    OK = "ok"           # Resources within safe limits
    PAUSE = "pause"     # Resources exceeded - pause orchestration
    RESUME = "resume"   # Resources recovered - resume orchestration


# =============================================================================
# RESOURCE GUARD
# =============================================================================

class ResourceGuard:
    """
    Background monitor for system resources (RAM, CPU).

    Prevents crashes by pausing orchestration when resources are constrained.

    Usage:
        # Load config
        config = load_config()

        # Start monitoring
        guard = ResourceGuard(config["system"]["resources"])
        guard.start()

        # Check before expensive operations
        if guard.get_signal() == ResourceSignal.PAUSE:
            guard.wait_for_resources(timeout=300)

        # Cleanup
        guard.stop()

    Configuration (from paragon.toml):
        ram_threshold_percent: RAM usage % to trigger pause (default: 90)
        cpu_threshold_percent: CPU usage % to trigger pause (default: 95)
        sustained_duration_seconds: Time threshold must be exceeded (default: 60)
        poll_interval_seconds: How often to check resources (default: 5)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the resource guard.

        Args:
            config: Configuration dict from [system.resources] section
        """
        # Load thresholds from config (no hardcoding)
        self.ram_threshold = config.get("ram_threshold_percent", 90)
        self.cpu_threshold = config.get("cpu_threshold_percent", 95)
        self.sustained_duration = config.get("sustained_duration_seconds", 60)
        self.poll_interval = config.get("poll_interval_seconds", 5)

        # Internal state
        self._signal = ResourceSignal.OK
        self._signal_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._resume_event = threading.Event()
        self._resume_event.set()  # Initially ready

        # Monitoring thread
        self._monitor_thread: Optional[threading.Thread] = None

        # Tracking for sustained violations
        self._violation_start_time: Optional[float] = None

        logger.info(
            f"ResourceGuard initialized: RAM<{self.ram_threshold}%, "
            f"CPU<{self.cpu_threshold}%, sustained={self.sustained_duration}s"
        )

    def start(self) -> None:
        """Start background resource monitoring."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("ResourceGuard already running")
            return

        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceGuard",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info("ResourceGuard monitoring started")

    def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_thread is None:
            return

        self._stop_event.set()
        self._monitor_thread.join(timeout=10)
        logger.info("ResourceGuard monitoring stopped")

    def get_signal(self) -> ResourceSignal:
        """
        Get the current resource signal.

        Returns:
            ResourceSignal.OK if resources available
            ResourceSignal.PAUSE if resources constrained
        """
        with self._signal_lock:
            return self._signal

    def wait_for_resources(self, timeout: Optional[float] = None) -> bool:
        """
        Block until resources become available.

        Args:
            timeout: Maximum seconds to wait (None = wait indefinitely)

        Returns:
            True if resources available, False if timeout
        """
        logger.warning(f"Waiting for resources to free up (timeout={timeout}s)...")
        result = self._resume_event.wait(timeout=timeout)

        if result:
            logger.info("Resources available - resuming")
        else:
            logger.error(f"Timeout waiting for resources after {timeout}s")

        return result

    def _monitor_loop(self) -> None:
        """
        Background monitoring loop.

        Strategy:
        1. Poll system resources every poll_interval seconds
        2. Track violations (RAM > threshold OR CPU > threshold)
        3. If violation sustained for sustained_duration, emit PAUSE
        4. Once resources recover, emit RESUME
        """
        logger.debug("ResourceGuard monitor loop started")

        while not self._stop_event.is_set():
            try:
                # Get current resource usage
                ram_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)

                # Check for violations
                ram_violation = ram_percent > self.ram_threshold
                cpu_violation = cpu_percent > self.cpu_threshold
                is_violation = ram_violation or cpu_violation

                current_time = time.time()

                if is_violation:
                    # Start tracking violation if new
                    if self._violation_start_time is None:
                        self._violation_start_time = current_time
                        logger.warning(
                            f"Resource constraint detected: RAM={ram_percent:.1f}%, "
                            f"CPU={cpu_percent:.1f}%"
                        )

                    # Check if sustained
                    violation_duration = current_time - self._violation_start_time

                    if violation_duration >= self.sustained_duration:
                        # Emit PAUSE signal
                        with self._signal_lock:
                            if self._signal != ResourceSignal.PAUSE:
                                self._signal = ResourceSignal.PAUSE
                                self._resume_event.clear()
                                logger.error(
                                    f"RESOURCE PAUSE triggered: RAM={ram_percent:.1f}%, "
                                    f"CPU={cpu_percent:.1f}% (sustained {violation_duration:.1f}s)"
                                )

                else:
                    # Resources OK - reset violation tracking
                    if self._violation_start_time is not None:
                        logger.info(
                            f"Resources recovered: RAM={ram_percent:.1f}%, "
                            f"CPU={cpu_percent:.1f}%"
                        )
                        self._violation_start_time = None

                    # Emit RESUME if we were paused
                    with self._signal_lock:
                        if self._signal == ResourceSignal.PAUSE:
                            self._signal = ResourceSignal.OK
                            self._resume_event.set()
                            logger.info("RESOURCE RESUME - orchestration can continue")

                # Sleep until next poll
                self._stop_event.wait(timeout=self.poll_interval)

            except Exception as e:
                logger.error(f"Error in resource monitor loop: {e}", exc_info=True)
                # Continue monitoring despite errors
                self._stop_event.wait(timeout=self.poll_interval)

        logger.debug("ResourceGuard monitor loop stopped")

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current resource usage (for debugging/telemetry).

        Returns:
            Dict with 'ram_percent' and 'cpu_percent'
        """
        return {
            "ram_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(interval=0.1),
        }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_guard_instance: Optional[ResourceGuard] = None


def get_resource_guard() -> Optional[ResourceGuard]:
    """
    Get the global ResourceGuard instance.

    Returns None if not initialized (call init_resource_guard first).
    """
    return _guard_instance


def init_resource_guard(config: Dict[str, Any]) -> ResourceGuard:
    """
    Initialize and start the global ResourceGuard.

    Args:
        config: Configuration dict from [system.resources] section

    Returns:
        The initialized ResourceGuard instance
    """
    global _guard_instance

    if _guard_instance is not None:
        logger.warning("ResourceGuard already initialized")
        return _guard_instance

    _guard_instance = ResourceGuard(config)
    _guard_instance.start()
    return _guard_instance


def shutdown_resource_guard() -> None:
    """Shutdown the global ResourceGuard (cleanup)."""
    global _guard_instance

    if _guard_instance is not None:
        _guard_instance.stop()
        _guard_instance = None
