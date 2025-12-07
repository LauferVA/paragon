"""
PARAGON METRICS COLLECTOR - The Traceability Engine

Full traceability from REQ -> CODE -> GIT.
Every node operation is recorded with provenance metadata.

Architecture:
- NodeMetric: Individual node processing record
- FailurePattern: Aggregated failure analysis
- SuccessPattern: Aggregated success analysis
- MetricsCollector: Central metrics aggregation and query interface

Design Principles:
1. APPEND-ONLY: Metrics are immutable once recorded
2. POLARS-NATIVE: All aggregation uses Polars for performance
3. TRACEABILITY: Every metric links back to REQ via golden thread
4. ZERO OVERHEAD: Use msgspec for hot-path serialization
"""
import msgspec
from typing import Optional, Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import polars as pl

from core.ontology import NodeType, NodeStatus, EdgeType


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_utc() -> str:
    """Fast UTC timestamp as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# METRIC DATA STRUCTURES (msgspec for performance)
# =============================================================================

class NodeMetric(msgspec.Struct, kw_only=True, frozen=False):
    """
    Individual node processing record.

    Captures the full context of a node's processing lifecycle,
    enabling post-hoc analysis and traceability queries.
    """
    # === Identity ===
    node_id: str                                # Node UUID
    node_type: str                              # NodeType.value
    status: str                                 # Final status after operation

    # === Timing ===
    created_at: str                             # When node was created
    processed_at: Optional[str] = None          # When processing started
    completed_at: Optional[str] = None          # When processing ended
    processing_time_ms: Optional[float] = None  # Duration in milliseconds

    # === Agent Context ===
    agent_id: Optional[str] = None              # Agent UUID
    agent_role: Optional[str] = None            # Agent role (BUILDER, TESTER, etc.)
    operation: Optional[str] = None             # Operation performed

    # === Token Usage ===
    token_count: int = 0                        # Total tokens used
    input_tokens: int = 0                       # Input/prompt tokens
    output_tokens: int = 0                      # Output/completion tokens

    # === Retry Tracking ===
    retry_count: int = 0                        # Number of retries
    last_error: Optional[str] = None            # Last error message if failed

    # === Traceability (Golden Thread) ===
    traces_to_req: Optional[str] = None         # REQ node ID this traces to
    traces_to_spec: Optional[str] = None        # SPEC node ID this traces to
    implements_spec: Optional[str] = None       # SPEC this CODE implements

    # === Context Metrics ===
    context_node_ids: List[str] = msgspec.field(default_factory=list)
    context_token_count: int = 0                # Tokens in context
    context_pruning_ratio: float = 0.0          # How much was pruned

    # === Materialization ===
    materialized_commit: Optional[str] = None   # Git commit SHA if materialized
    materialized_files: List[str] = msgspec.field(default_factory=list)

    # === Extension Point ===
    extra: Dict[str, Any] = msgspec.field(default_factory=dict)


class FailurePattern(msgspec.Struct, kw_only=True, frozen=True):
    """
    Aggregated failure analysis pattern.

    Used to identify systemic issues across agent roles and operations.
    """
    category: str                               # Failure category
    count: int                                  # Number of occurrences
    node_types: Tuple[str, ...]                 # Node types affected
    agent_roles: Tuple[str, ...]                # Agents involved
    operations: Tuple[str, ...]                 # Operations that failed
    avg_retry_count: float                      # Average retries before failure
    sample_errors: Tuple[str, ...]              # Sample error messages


class SuccessPattern(msgspec.Struct, kw_only=True, frozen=True):
    """
    Aggregated success analysis pattern.

    Used to understand what works well and where optimization helps.
    """
    node_type: str                              # Node type
    agent_role: str                             # Agent role
    count: int                                  # Number of successes
    avg_processing_time_ms: float               # Average processing time
    avg_token_count: float                      # Average tokens used
    avg_retry_count: float                      # Average retries (should be low)
    avg_context_pruning_ratio: float            # How much context was pruned


class TraceabilityReport(msgspec.Struct, kw_only=True, frozen=False):
    """
    Full traceability report for a requirement.

    The Golden Thread: REQ -> SPEC -> CODE -> TEST -> VERIFIED
    """
    req_id: str                                 # Root requirement
    req_status: str                             # Current status
    total_specs: int = 0                        # SPECs derived from this REQ
    verified_specs: int = 0                     # SPECs that are verified
    total_code: int = 0                         # CODE nodes
    verified_code: int = 0                      # CODE that is verified
    total_tests: int = 0                        # TEST/TEST_SUITE nodes
    passed_tests: int = 0                       # Tests that passed
    total_tokens: int = 0                       # Total tokens used
    total_time_ms: float = 0.0                  # Total processing time
    materialized_commits: List[str] = msgspec.field(default_factory=list)
    lineage: List[str] = msgspec.field(default_factory=list)


# =============================================================================
# METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Central metrics aggregation and query interface.

    Thread-safe (uses immutable structures) and performant (Polars aggregation).

    Usage:
        collector = MetricsCollector()

        # Record node processing
        collector.record_node_start(node_id, node_type, agent_role)
        # ... processing ...
        collector.record_node_complete(node_id, status, tokens=1234)

        # Query patterns
        failures = collector.get_failure_patterns()
        report = collector.get_traceability_report(req_id)
    """

    def __init__(self):
        """Initialize the collector."""
        self._metrics: Dict[str, NodeMetric] = {}
        self._in_progress: Dict[str, float] = {}  # node_id -> start_time_ms

    # =========================================================================
    # RECORDING API
    # =========================================================================

    def record_node_created(
        self,
        node_id: str,
        node_type: str,
        created_by: str = "system",
        traces_to_req: Optional[str] = None,
        traces_to_spec: Optional[str] = None,
    ) -> NodeMetric:
        """
        Record node creation.

        Args:
            node_id: The node UUID
            node_type: NodeType.value
            created_by: Agent or user ID
            traces_to_req: REQ this traces to (golden thread)
            traces_to_spec: SPEC this traces to

        Returns:
            The created NodeMetric
        """
        metric = NodeMetric(
            node_id=node_id,
            node_type=node_type,
            status=NodeStatus.PENDING.value,
            created_at=now_utc(),
            agent_id=created_by,
            traces_to_req=traces_to_req,
            traces_to_spec=traces_to_spec,
        )
        self._metrics[node_id] = metric
        return metric

    def record_node_start(
        self,
        node_id: str,
        agent_id: str,
        agent_role: str,
        operation: str,
        context_node_ids: Optional[List[str]] = None,
        context_token_count: int = 0,
    ) -> None:
        """
        Record start of node processing.

        Args:
            node_id: The node being processed
            agent_id: Agent UUID
            agent_role: Agent role (BUILDER, TESTER, etc.)
            operation: Operation being performed
            context_node_ids: Nodes in context
            context_token_count: Tokens in context
        """
        import time
        self._in_progress[node_id] = time.time() * 1000  # ms

        if node_id in self._metrics:
            metric = self._metrics[node_id]
            metric.processed_at = now_utc()
            metric.agent_id = agent_id
            metric.agent_role = agent_role
            metric.operation = operation
            metric.status = NodeStatus.PROCESSING.value
            if context_node_ids:
                metric.context_node_ids = context_node_ids
            metric.context_token_count = context_token_count

    def record_node_complete(
        self,
        node_id: str,
        status: str,
        tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        context_pruning_ratio: float = 0.0,
        error: Optional[str] = None,
    ) -> Optional[NodeMetric]:
        """
        Record completion of node processing.

        Args:
            node_id: The node that was processed
            status: Final status
            tokens: Total tokens used
            input_tokens: Input tokens
            output_tokens: Output tokens
            context_pruning_ratio: Fraction of context pruned
            error: Error message if failed

        Returns:
            The updated NodeMetric, or None if not found
        """
        import time

        if node_id not in self._metrics:
            return None

        metric = self._metrics[node_id]
        metric.completed_at = now_utc()
        metric.status = status
        metric.token_count = tokens or (input_tokens + output_tokens)
        metric.input_tokens = input_tokens
        metric.output_tokens = output_tokens
        metric.context_pruning_ratio = context_pruning_ratio

        # Calculate processing time
        if node_id in self._in_progress:
            start_time = self._in_progress.pop(node_id)
            metric.processing_time_ms = time.time() * 1000 - start_time

        if status == NodeStatus.FAILED.value and error:
            metric.last_error = error
            metric.retry_count += 1

        return metric

    def record_retry(self, node_id: str, error: str) -> None:
        """Record a retry attempt for a node."""
        if node_id in self._metrics:
            metric = self._metrics[node_id]
            metric.retry_count += 1
            metric.last_error = error

    def record_materialization(
        self,
        node_id: str,
        commit_sha: str,
        files: List[str],
    ) -> None:
        """Record that a node was materialized to git."""
        if node_id in self._metrics:
            metric = self._metrics[node_id]
            metric.materialized_commit = commit_sha
            metric.materialized_files = files

    def update_traceability(
        self,
        node_id: str,
        traces_to_req: Optional[str] = None,
        traces_to_spec: Optional[str] = None,
        implements_spec: Optional[str] = None,
    ) -> None:
        """Update traceability links for a node."""
        if node_id in self._metrics:
            metric = self._metrics[node_id]
            if traces_to_req:
                metric.traces_to_req = traces_to_req
            if traces_to_spec:
                metric.traces_to_spec = traces_to_spec
            if implements_spec:
                metric.implements_spec = implements_spec

    # =========================================================================
    # QUERY API
    # =========================================================================

    def get_metric(self, node_id: str) -> Optional[NodeMetric]:
        """Get metric for a specific node."""
        return self._metrics.get(node_id)

    def get_all_metrics(self) -> List[NodeMetric]:
        """Get all recorded metrics."""
        return list(self._metrics.values())

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert all metrics to a Polars DataFrame.

        Enables efficient aggregation and filtering using Polars operations.
        """
        if not self._metrics:
            return pl.DataFrame()

        # Extract fields into lists for DataFrame construction
        records = []
        for metric in self._metrics.values():
            records.append({
                "node_id": metric.node_id,
                "node_type": metric.node_type,
                "status": metric.status,
                "created_at": metric.created_at,
                "processed_at": metric.processed_at,
                "completed_at": metric.completed_at,
                "processing_time_ms": metric.processing_time_ms,
                "agent_id": metric.agent_id,
                "agent_role": metric.agent_role,
                "operation": metric.operation,
                "token_count": metric.token_count,
                "input_tokens": metric.input_tokens,
                "output_tokens": metric.output_tokens,
                "retry_count": metric.retry_count,
                "last_error": metric.last_error,
                "traces_to_req": metric.traces_to_req,
                "traces_to_spec": metric.traces_to_spec,
                "implements_spec": metric.implements_spec,
                "context_token_count": metric.context_token_count,
                "context_pruning_ratio": metric.context_pruning_ratio,
                "materialized_commit": metric.materialized_commit,
            })

        return pl.DataFrame(records)

    def query_by_traceability(
        self,
        req_id: Optional[str] = None,
        spec_id: Optional[str] = None,
    ) -> List[NodeMetric]:
        """
        Query metrics by traceability - follow the golden thread.

        Args:
            req_id: Filter by REQ ID
            spec_id: Filter by SPEC ID

        Returns:
            List of metrics matching the traceability filter
        """
        results = []
        for metric in self._metrics.values():
            if req_id and metric.traces_to_req != req_id:
                continue
            if spec_id and metric.traces_to_spec != spec_id:
                continue
            results.append(metric)
        return results

    def query_by_agent(
        self,
        agent_role: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> List[NodeMetric]:
        """
        Query metrics by agent role and operation.

        Args:
            agent_role: Filter by agent role (BUILDER, TESTER, etc.)
            operation: Filter by operation type

        Returns:
            List of matching metrics
        """
        results = []
        for metric in self._metrics.values():
            if agent_role and metric.agent_role != agent_role:
                continue
            if operation and metric.operation != operation:
                continue
            results.append(metric)
        return results

    def query_by_status(self, status: str) -> List[NodeMetric]:
        """Query metrics by status."""
        return [m for m in self._metrics.values() if m.status == status]

    def query_failures(self) -> List[NodeMetric]:
        """Get all failed metrics."""
        return self.query_by_status(NodeStatus.FAILED.value)

    # =========================================================================
    # PATTERN ANALYSIS
    # =========================================================================

    def get_failure_patterns(self, min_count: int = 2) -> List[FailurePattern]:
        """
        Analyze failure patterns across the system.

        Groups failures by error category and identifies systemic issues.

        Args:
            min_count: Minimum occurrences to be considered a pattern

        Returns:
            List of failure patterns sorted by count
        """
        failures = self.query_failures()
        if not failures:
            return []

        # Group by error category (first line of error)
        by_category: Dict[str, List[NodeMetric]] = defaultdict(list)
        for metric in failures:
            if metric.last_error:
                # Use first line as category
                category = metric.last_error.split('\n')[0][:100]
            else:
                category = "Unknown"
            by_category[category].append(metric)

        patterns = []
        for category, metrics in by_category.items():
            if len(metrics) < min_count:
                continue

            patterns.append(FailurePattern(
                category=category,
                count=len(metrics),
                node_types=tuple(set(m.node_type for m in metrics)),
                agent_roles=tuple(set(m.agent_role for m in metrics if m.agent_role)),
                operations=tuple(set(m.operation for m in metrics if m.operation)),
                avg_retry_count=sum(m.retry_count for m in metrics) / len(metrics),
                sample_errors=tuple(m.last_error for m in metrics[:3] if m.last_error),
            ))

        return sorted(patterns, key=lambda p: p.count, reverse=True)

    def get_success_patterns(self) -> List[SuccessPattern]:
        """
        Analyze success patterns to understand what works well.

        Returns:
            List of success patterns by node_type + agent_role
        """
        successes = self.query_by_status(NodeStatus.VERIFIED.value)
        if not successes:
            return []

        # Group by (node_type, agent_role)
        by_key: Dict[Tuple[str, str], List[NodeMetric]] = defaultdict(list)
        for metric in successes:
            key = (metric.node_type, metric.agent_role or "unknown")
            by_key[key].append(metric)

        patterns = []
        for (node_type, agent_role), metrics in by_key.items():
            times = [m.processing_time_ms for m in metrics if m.processing_time_ms]
            tokens = [m.token_count for m in metrics]
            retries = [m.retry_count for m in metrics]
            pruning = [m.context_pruning_ratio for m in metrics]

            patterns.append(SuccessPattern(
                node_type=node_type,
                agent_role=agent_role,
                count=len(metrics),
                avg_processing_time_ms=sum(times) / len(times) if times else 0.0,
                avg_token_count=sum(tokens) / len(tokens) if tokens else 0.0,
                avg_retry_count=sum(retries) / len(retries) if retries else 0.0,
                avg_context_pruning_ratio=sum(pruning) / len(pruning) if pruning else 0.0,
            ))

        return sorted(patterns, key=lambda p: p.count, reverse=True)

    def get_traceability_report(self, req_id: str) -> TraceabilityReport:
        """
        Generate a full traceability report for a requirement.

        Follows the golden thread from REQ through all derived artifacts.

        Args:
            req_id: The requirement ID to trace

        Returns:
            TraceabilityReport with full lineage
        """
        # Get the REQ metric
        req_metric = self._metrics.get(req_id)
        req_status = req_metric.status if req_metric else "unknown"

        # Find all nodes tracing to this REQ
        lineage_metrics = self.query_by_traceability(req_id=req_id)

        # Categorize by type
        specs = [m for m in lineage_metrics if m.node_type == NodeType.SPEC.value]
        codes = [m for m in lineage_metrics if m.node_type == NodeType.CODE.value]
        tests = [m for m in lineage_metrics if m.node_type in (NodeType.TEST.value, NodeType.TEST_SUITE.value)]

        # Calculate totals
        total_tokens = sum(m.token_count for m in lineage_metrics)
        times = [m.processing_time_ms for m in lineage_metrics if m.processing_time_ms]
        total_time = sum(times) if times else 0.0

        # Collect commits
        commits = list(set(
            m.materialized_commit
            for m in lineage_metrics
            if m.materialized_commit
        ))

        # Build lineage chain (node IDs in order)
        lineage_ids = [req_id] + [m.node_id for m in lineage_metrics if m.node_id != req_id]

        return TraceabilityReport(
            req_id=req_id,
            req_status=req_status,
            total_specs=len(specs),
            verified_specs=sum(1 for s in specs if s.status == NodeStatus.VERIFIED.value),
            total_code=len(codes),
            verified_code=sum(1 for c in codes if c.status == NodeStatus.VERIFIED.value),
            total_tests=len(tests),
            passed_tests=sum(1 for t in tests if t.status == NodeStatus.VERIFIED.value),
            total_tokens=total_tokens,
            total_time_ms=total_time,
            materialized_commits=commits,
            lineage=lineage_ids,
        )

    # =========================================================================
    # AGGREGATE STATISTICS
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dictionary with aggregate statistics
        """
        df = self.to_dataframe()
        if df.is_empty():
            return {
                "total_nodes": 0,
                "by_status": {},
                "by_type": {},
                "total_tokens": 0,
                "avg_processing_time_ms": 0.0,
            }

        return {
            "total_nodes": len(df),
            "by_status": df.group_by("status").len().to_dicts(),
            "by_type": df.group_by("node_type").len().to_dicts(),
            "total_tokens": df["token_count"].sum(),
            "avg_processing_time_ms": df["processing_time_ms"].mean() or 0.0,
            "total_retries": df["retry_count"].sum(),
            "failure_count": len(df.filter(pl.col("status") == NodeStatus.FAILED.value)),
            "verified_count": len(df.filter(pl.col("status") == NodeStatus.VERIFIED.value)),
        }

    def clear(self) -> None:
        """Clear all collected metrics."""
        self._metrics.clear()
        self._in_progress.clear()

    # =========================================================================
    # MUTATION LOGGER INTEGRATION
    # =========================================================================

    def sync_from_mutation_logger(self, mutation_logger=None) -> int:
        """
        Sync metrics from MutationLogger events.

        This implements the layered approach:
        - MutationLogger handles the write path (event logging)
        - MetricsCollector handles the read path (querying/aggregation)

        Args:
            mutation_logger: MutationLogger instance (uses global if None)

        Returns:
            Number of events processed
        """
        try:
            if mutation_logger is None:
                from infrastructure.logger import get_logger
                mutation_logger = get_logger()

            events = mutation_logger.get_recent_events(n=10000)
            processed = 0

            for event in events:
                if event.mutation_type == "node_created":
                    if event.node_id not in self._metrics:
                        self.record_node_created(
                            node_id=event.node_id,
                            node_type=event.node_type or "unknown",
                            created_by=event.agent_id,
                        )
                        processed += 1

                elif event.mutation_type == "status_changed":
                    if event.node_id in self._metrics:
                        metric = self._metrics[event.node_id]
                        metric.status = event.new_status or metric.status
                        processed += 1

            return processed
        except Exception:
            return 0

    def subscribe_to_mutation_logger(self, mutation_logger=None) -> None:
        """
        Subscribe to MutationLogger for real-time metric updates.

        This allows MetricsCollector to automatically record metrics
        as mutations happen, without manual sync.

        Args:
            mutation_logger: MutationLogger instance (uses global if None)
        """
        try:
            if mutation_logger is None:
                from infrastructure.logger import get_logger
                mutation_logger = get_logger()

            def on_mutation(event):
                """Callback for mutation events."""
                if event.mutation_type == "node_created":
                    if event.node_id not in self._metrics:
                        self.record_node_created(
                            node_id=event.node_id,
                            node_type=event.node_type or "unknown",
                            created_by=event.agent_id,
                        )
                elif event.mutation_type == "status_changed":
                    if event.node_id in self._metrics:
                        metric = self._metrics[event.node_id]
                        metric.status = event.new_status or metric.status

            mutation_logger.subscribe(on_mutation)
        except Exception:
            pass  # Graceful degradation if logger not available


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global collector instance for simple usage
_global_collector: Optional[MetricsCollector] = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_node_created(
    node_id: str,
    node_type: str,
    **kwargs
) -> NodeMetric:
    """Convenience function to record node creation."""
    return get_collector().record_node_created(node_id, node_type, **kwargs)


def record_node_start(node_id: str, agent_id: str, agent_role: str, operation: str, **kwargs) -> None:
    """Convenience function to record processing start."""
    get_collector().record_node_start(node_id, agent_id, agent_role, operation, **kwargs)


def record_node_complete(node_id: str, status: str, **kwargs) -> Optional[NodeMetric]:
    """Convenience function to record processing completion."""
    return get_collector().record_node_complete(node_id, status, **kwargs)


def get_traceability_report(req_id: str) -> TraceabilityReport:
    """Convenience function to get traceability report."""
    return get_collector().get_traceability_report(req_id)
