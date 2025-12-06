"""
PARAGON DIAGNOSTICS - System State & Performance Tracking

Fast diagnosis of runtime issues:
- Global state (DB, LLM, RateLimiter)
- LLM call metrics (timing, tokens, retries, truncation)
- TDD cycle phase timing
- Rate limit status

Outputs clear [CLEAN] / [BLOAT] / [WARN] markers for rapid troubleshooting.

Correlation IDs link diagnostic events to mutation logs:
- Each session gets a unique correlation_id
- Pass correlation_id to MutationLogger for cross-referencing
- Query both logs by correlation_id to see full picture

Usage:
    from infrastructure.diagnostics import diag, print_state_summary

    # At start of test/run
    dx = diag()
    dx.set_session("my-session")  # Sets correlation_id
    print_state_summary()

    # Wrap LLM calls
    with dx.llm_call("ImplementationPlan") as call:
        result = llm.generate(...)
        call.set_tokens(result.usage.prompt_tokens, result.usage.completion_tokens)

    # Link to mutation log
    from infrastructure.logger import log_node_created
    log_node_created("node_1", "CODE", correlation_id=dx.correlation_id)

    # At end
    dx.print_summary()
"""
import time
import logging
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def generate_correlation_id() -> str:
    """Generate a unique correlation ID for linking logs."""
    return f"dx_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


# =============================================================================
# STATE MARKERS
# =============================================================================

class StateMarker:
    """Visual markers for diagnostic output."""
    CLEAN = "\033[92m[CLEAN]\033[0m"      # Green
    BLOAT = "\033[93m[BLOAT]\033[0m"      # Yellow
    WARN = "\033[91m[WARN]\033[0m"        # Red
    INFO = "\033[94m[INFO]\033[0m"        # Blue

    # Non-colored versions for logs
    CLEAN_PLAIN = "[CLEAN]"
    BLOAT_PLAIN = "[BLOAT]"
    WARN_PLAIN = "[WARN]"
    INFO_PLAIN = "[INFO]"


# =============================================================================
# LLM CALL METRICS
# =============================================================================

@dataclass
class LLMCallMetric:
    """Metrics for a single LLM call."""
    schema_name: str
    start_time: float
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    truncated: bool = False

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema_name,
            "duration_ms": round(self.duration_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "success": self.success,
            "error": self.error,
            "retry_count": self.retry_count,
            "truncated": self.truncated,
        }


@dataclass
class LLMCallContext:
    """Context manager for tracking an LLM call."""
    metric: LLMCallMetric
    diagnostics: "DiagnosticLogger"

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.metric.input_tokens = input_tokens
        self.metric.output_tokens = output_tokens

    def set_error(self, error: str) -> None:
        self.metric.error = error
        self.metric.success = False

    def set_truncated(self, truncated: bool = True) -> None:
        self.metric.truncated = truncated

    def set_retry_count(self, count: int) -> None:
        self.metric.retry_count = count

    def __enter__(self) -> "LLMCallContext":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.metric.end_time = time.time()
        if exc_type is None:
            self.metric.success = True
        else:
            self.metric.error = str(exc_val)
            self.metric.success = False
        self.diagnostics._record_llm_call(self.metric)


# =============================================================================
# TDD PHASE TIMING
# =============================================================================

@dataclass
class PhaseMetric:
    """Metrics for a TDD phase."""
    phase_name: str
    start_time: float
    end_time: Optional[float] = None
    llm_calls: int = 0
    nodes_created: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000


# =============================================================================
# DIAGNOSTIC LOGGER
# =============================================================================

class DiagnosticLogger:
    """
    System state and performance diagnostics.

    Tracks:
    - Global state snapshots (DB, LLM, RateLimiter)
    - LLM call metrics (timing, tokens, retries)
    - TDD phase timing
    - Rate limit status

    Correlation:
    - Each session gets a unique correlation_id
    - Pass correlation_id to MutationLogger events
    - Query both logs by correlation_id to see full picture
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or Path("workspace/logs/diagnostics.jsonl")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._llm_calls: List[LLMCallMetric] = []
        self._phase_metrics: List[PhaseMetric] = []
        self._current_phase: Optional[PhaseMetric] = None
        self._session_start = time.time()
        self._session_id: Optional[str] = None
        self._correlation_id: Optional[str] = None

    @property
    def correlation_id(self) -> Optional[str]:
        """Get the current correlation ID for linking to mutation logs."""
        return self._correlation_id

    def set_session(self, session_id: str) -> str:
        """
        Set session ID and generate correlation ID.

        Returns:
            The generated correlation_id for linking to mutation logs
        """
        self._session_id = session_id
        self._correlation_id = generate_correlation_id()
        self._session_start = time.time()
        self._llm_calls = []
        self._phase_metrics = []

        # Log session start
        self._write_log("session_start", {
            "correlation_id": self._correlation_id,
        })

        logger.info(f"[DIAG] Session started: {session_id} (correlation={self._correlation_id})")
        return self._correlation_id

    # =========================================================================
    # STATE SNAPSHOTS
    # =========================================================================

    def get_db_state(self) -> Dict[str, Any]:
        """Get current database state."""
        try:
            from agents.tools import get_db
            db = get_db()
            if db is None:
                return {"status": "CLEAN", "message": "Global DB is None"}
            return {
                "status": "ACTIVE",
                "node_count": db.node_count,
                "edge_count": db.edge_count,
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def get_llm_state(self) -> Dict[str, Any]:
        """Get current LLM instance state."""
        try:
            from core.llm import _llm_instance
            if _llm_instance is None:
                return {"status": "CLEAN", "message": "LLM Instance is None"}
            return {
                "status": "ACTIVE",
                "model": _llm_instance.model,
                "temperature": _llm_instance.temperature,
                "max_tokens": _llm_instance.max_tokens,
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def get_rate_limit_state(self) -> Dict[str, Any]:
        """Get current rate limiter state."""
        try:
            from core.llm import _rate_limit_guard
            if _rate_limit_guard is None:
                return {"status": "CLEAN", "message": "RateLimiter is None"}
            status = _rate_limit_guard.get_status()
            return {
                "status": "ACTIVE",
                "rpm_used": status["rpm_used"],
                "rpm_limit": status["rpm_limit"],
                "tpm_used": status["tpm_used"],
                "tpm_limit": status["tpm_limit"],
                "retry_after": status["retry_after_remaining"],
            }
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def print_state_summary(self, use_color: bool = True) -> None:
        """Print current system state summary."""
        M = StateMarker if use_color else type('M', (), {
            'CLEAN': '[CLEAN]', 'BLOAT': '[BLOAT]',
            'WARN': '[WARN]', 'INFO': '[INFO]'
        })()

        print("\n" + "=" * 60)
        print("PARAGON DIAGNOSTICS - State Summary")
        print("=" * 60)

        # DB State
        db = self.get_db_state()
        if db["status"] == "CLEAN":
            print(f"{M.CLEAN} Global DB is None")
        elif db["status"] == "ACTIVE":
            count = db["node_count"]
            marker = M.BLOAT if count > 100 else M.INFO
            print(f"{marker} Global DB Active. Nodes={count}, Edges={db['edge_count']}")
        else:
            print(f"{M.WARN} DB Error: {db['message']}")

        # LLM State
        llm = self.get_llm_state()
        if llm["status"] == "CLEAN":
            print(f"{M.CLEAN} LLM Instance is None")
        elif llm["status"] == "ACTIVE":
            print(f"{M.INFO} LLM Active. Model={llm['model']}, MaxTokens={llm['max_tokens']}")
        else:
            print(f"{M.WARN} LLM Error: {llm['message']}")

        # Rate Limiter State
        rl = self.get_rate_limit_state()
        if rl["status"] == "CLEAN":
            print(f"{M.CLEAN} RateLimiter is None")
        elif rl["status"] == "ACTIVE":
            rpm_pct = (rl["rpm_used"] / rl["rpm_limit"]) * 100 if rl["rpm_limit"] > 0 else 0
            tpm_pct = (rl["tpm_used"] / rl["tpm_limit"]) * 100 if rl["tpm_limit"] > 0 else 0
            marker = M.WARN if rpm_pct > 80 or tpm_pct > 80 else M.INFO
            print(f"{marker} RateLimiter. RPM={rl['rpm_used']}/{rl['rpm_limit']} ({rpm_pct:.0f}%), TPM={rl['tpm_used']}/{rl['tpm_limit']} ({tpm_pct:.0f}%)")
            if rl["retry_after"] > 0:
                print(f"{M.WARN} Retry-After active: {rl['retry_after']:.1f}s remaining")
        else:
            print(f"{M.WARN} RateLimiter Error: {rl['message']}")

        print("=" * 60 + "\n")

    # =========================================================================
    # LLM CALL TRACKING
    # =========================================================================

    def llm_call(self, schema_name: str) -> LLMCallContext:
        """
        Context manager for tracking an LLM call.

        Usage:
            with diag.llm_call("ImplementationPlan") as call:
                result = llm.generate(...)
                call.set_tokens(result.usage.prompt_tokens, result.usage.completion_tokens)
        """
        metric = LLMCallMetric(
            schema_name=schema_name,
            start_time=time.time(),
        )
        return LLMCallContext(metric=metric, diagnostics=self)

    def _record_llm_call(self, metric: LLMCallMetric) -> None:
        """Record a completed LLM call."""
        self._llm_calls.append(metric)

        # Log immediately
        status = "OK" if metric.success else "FAIL"
        trunc = " [TRUNCATED]" if metric.truncated else ""
        retry = f" [RETRY x{metric.retry_count}]" if metric.retry_count > 0 else ""

        msg = (
            f"[LLM] {metric.schema_name}: {metric.duration_ms:.0f}ms, "
            f"in={metric.input_tokens}, out={metric.output_tokens} "
            f"[{status}]{trunc}{retry}"
        )

        if metric.success:
            logger.info(msg)
        else:
            logger.warning(f"{msg} - {metric.error}")

        # Write to file
        self._write_log("llm_call", metric.to_dict())

    def record_llm_call_simple(
        self,
        schema_name: str,
        duration_ms: float,
        input_tokens: int,
        output_tokens: int,
        success: bool = True,
        error: Optional[str] = None,
        truncated: bool = False,
        retry_count: int = 0,
    ) -> None:
        """Record an LLM call without context manager."""
        metric = LLMCallMetric(
            schema_name=schema_name,
            start_time=time.time() - (duration_ms / 1000),
            end_time=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error=error,
            truncated=truncated,
            retry_count=retry_count,
        )
        self._record_llm_call(metric)

    # =========================================================================
    # TDD PHASE TRACKING
    # =========================================================================

    def start_phase(self, phase_name: str) -> None:
        """Start tracking a TDD phase."""
        if self._current_phase is not None:
            self.end_phase()

        self._current_phase = PhaseMetric(
            phase_name=phase_name,
            start_time=time.time(),
        )
        logger.info(f"[PHASE] Starting: {phase_name}")

    def end_phase(self, success: bool = True, error: Optional[str] = None) -> None:
        """End tracking the current phase."""
        if self._current_phase is None:
            return

        self._current_phase.end_time = time.time()
        self._current_phase.success = success
        self._current_phase.error = error

        status = "OK" if success else "FAIL"
        msg = f"[PHASE] {self._current_phase.phase_name}: {self._current_phase.duration_ms:.0f}ms [{status}]"

        if success:
            logger.info(msg)
        else:
            logger.warning(f"{msg} - {error}")

        self._phase_metrics.append(self._current_phase)
        self._write_log("phase", {
            "phase": self._current_phase.phase_name,
            "duration_ms": round(self._current_phase.duration_ms, 1),
            "success": success,
            "error": error,
        })
        self._current_phase = None

    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for phase tracking."""
        self.start_phase(phase_name)
        try:
            yield
            self.end_phase(success=True)
        except Exception as e:
            self.end_phase(success=False, error=str(e))
            raise

    # =========================================================================
    # SUMMARY & LOGGING
    # =========================================================================

    def print_summary(self, use_color: bool = True) -> None:
        """Print session summary."""
        M = StateMarker if use_color else type('M', (), {
            'CLEAN': '[CLEAN]', 'BLOAT': '[BLOAT]',
            'WARN': '[WARN]', 'INFO': '[INFO]'
        })()

        total_duration = (time.time() - self._session_start) * 1000

        print("\n" + "=" * 60)
        print("PARAGON DIAGNOSTICS - Session Summary")
        print("=" * 60)

        if self._session_id:
            print(f"Session: {self._session_id}")
        print(f"Duration: {total_duration:.0f}ms ({total_duration/1000:.1f}s)")

        # LLM Call Summary
        if self._llm_calls:
            print(f"\nLLM Calls: {len(self._llm_calls)}")
            total_input = sum(c.input_tokens for c in self._llm_calls)
            total_output = sum(c.output_tokens for c in self._llm_calls)
            total_llm_time = sum(c.duration_ms for c in self._llm_calls)
            failed = sum(1 for c in self._llm_calls if not c.success)
            truncated = sum(1 for c in self._llm_calls if c.truncated)
            retried = sum(1 for c in self._llm_calls if c.retry_count > 0)

            print(f"  Total tokens: {total_input} in, {total_output} out")
            print(f"  Total LLM time: {total_llm_time:.0f}ms")

            if failed > 0:
                print(f"  {M.WARN} Failed: {failed}")
            if truncated > 0:
                print(f"  {M.WARN} Truncated: {truncated}")
            if retried > 0:
                print(f"  {M.WARN} Retried: {retried}")

            # Per-schema breakdown
            schemas: Dict[str, List[LLMCallMetric]] = {}
            for c in self._llm_calls:
                schemas.setdefault(c.schema_name, []).append(c)

            print("\n  Per Schema:")
            for schema, calls in schemas.items():
                avg_time = sum(c.duration_ms for c in calls) / len(calls)
                avg_tokens = sum(c.input_tokens + c.output_tokens for c in calls) / len(calls)
                print(f"    {schema}: {len(calls)} calls, avg {avg_time:.0f}ms, avg {avg_tokens:.0f} tokens")

        # Phase Summary
        if self._phase_metrics:
            print(f"\nPhases: {len(self._phase_metrics)}")
            for phase in self._phase_metrics:
                status = "OK" if phase.success else "FAIL"
                marker = M.INFO if phase.success else M.WARN
                print(f"  {marker} {phase.phase_name}: {phase.duration_ms:.0f}ms [{status}]")

        # Final state
        print("\nFinal State:")
        db = self.get_db_state()
        if db["status"] == "ACTIVE":
            print(f"  DB: {db['node_count']} nodes, {db['edge_count']} edges")

        rl = self.get_rate_limit_state()
        if rl["status"] == "ACTIVE":
            print(f"  RateLimiter: {rl['rpm_used']}/{rl['rpm_limit']} RPM, {rl['tpm_used']}/{rl['tpm_limit']} TPM")

        print("=" * 60 + "\n")

    def _write_log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a log entry to the diagnostics file."""
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": self._session_id,
                "correlation_id": self._correlation_id,  # Links to mutation logs
                "type": event_type,
                **data,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write diagnostic log: {e}")

    def reset(self) -> None:
        """Reset all metrics."""
        self._llm_calls = []
        self._phase_metrics = []
        self._current_phase = None
        self._session_start = time.time()


# =============================================================================
# GLOBAL INSTANCE & CONVENIENCE FUNCTIONS
# =============================================================================

_diagnostics: Optional[DiagnosticLogger] = None


def get_diagnostics() -> DiagnosticLogger:
    """Get or create the global diagnostics instance."""
    global _diagnostics
    if _diagnostics is None:
        _diagnostics = DiagnosticLogger()
    return _diagnostics


def reset_diagnostics() -> None:
    """Reset the global diagnostics instance."""
    global _diagnostics
    if _diagnostics is not None:
        _diagnostics.reset()
    _diagnostics = None


def print_state_summary(use_color: bool = True) -> None:
    """Print current system state summary."""
    get_diagnostics().print_state_summary(use_color)


def print_session_summary(use_color: bool = True) -> None:
    """Print session summary."""
    get_diagnostics().print_summary(use_color)


# Convenience alias
diag = get_diagnostics
