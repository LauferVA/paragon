"""
PARAGON BENCHMARK HARNESS

Executes golden set tasks against the Paragon system with:
- Isolated workspace per run (overwritten on re-run)
- Comprehensive metrics collection
- Rerun visualization integration
- Research/Dialectic interaction support
- Automatic cleanup of artifacts

Usage:
    python -m benchmarks.harness --tier smoke
    python -m benchmarks.harness --tier core
    python -m benchmarks.harness --task builder_001
    python -m benchmarks.harness --tier full --interactive

Folder Structure (all under workspace/benchmark/):
    runs/{run_id}/          - Current run artifacts (overwritten)
    metrics/{run_id}.jsonl  - Metrics for each run
    visualizations/         - Rerun .rrd files
"""
import sys
import os
import shutil
import json
import time
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add paragon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.graph_db import ParagonDB
from core.llm import get_llm, reset_llm, get_rate_limit_guard
from agents.orchestrator import TDDOrchestrator, run_tdd_cycle
from agents.tools import set_db, get_db, get_graph_stats
from infrastructure.diagnostics import get_diagnostics, reset_diagnostics

# Optional imports
try:
    from infrastructure.rerun_logger import RerunLogger, create_logger as create_rerun_logger
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    RerunLogger = None

try:
    from agents.research import create_research_graph
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

WORKSPACE_ROOT = Path(__file__).parent.parent / "workspace" / "benchmark"
RUNS_DIR = WORKSPACE_ROOT / "runs"
METRICS_DIR = WORKSPACE_ROOT / "metrics"
VIZ_DIR = WORKSPACE_ROOT / "visualizations"
GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.yaml"


# =============================================================================
# METRICS COLLECTION
# =============================================================================

@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str
    task_name: str
    task_type: str

    # Timing
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0

    # Outcome
    success: bool = False
    final_status: str = ""
    error_message: str = ""

    # Graph metrics
    nodes_created: int = 0
    edges_created: int = 0
    node_types: Dict[str, int] = field(default_factory=dict)
    edge_types: Dict[str, int] = field(default_factory=dict)
    wave_count: int = 0

    # LLM metrics
    llm_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_time_ms: float = 0.0
    schemas_used: Dict[str, int] = field(default_factory=dict)

    # TDD cycle metrics
    iterations: int = 0
    phases_completed: List[str] = field(default_factory=list)

    # Cost
    estimated_cost_usd: float = 0.0

    # Files generated
    files_created: List[str] = field(default_factory=list)

    # Research/Dialectic
    research_conducted: bool = False
    user_questions_asked: int = 0
    ambiguities_found: int = 0


@dataclass
class RunMetrics:
    """Metrics for a complete benchmark run."""
    run_id: str
    tier: str
    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0

    # Summary
    tasks_attempted: int = 0
    tasks_passed: int = 0
    tasks_failed: int = 0

    # Aggregates
    total_nodes: int = 0
    total_edges: int = 0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Per-task metrics
    task_metrics: List[TaskMetrics] = field(default_factory=list)

    # Insights
    slowest_task: str = ""
    most_expensive_task: str = ""
    failure_reasons: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collects and persists metrics during benchmark runs."""

    def __init__(self, run_id: str, tier: str):
        self.run_id = run_id
        self.tier = tier
        self.run_metrics = RunMetrics(
            run_id=run_id,
            tier=tier,
            start_time=datetime.now(timezone.utc).isoformat(),
        )
        self.current_task: Optional[TaskMetrics] = None

        # Ensure directories exist
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def start_task(self, task_id: str, task_name: str, task_type: str) -> None:
        """Start tracking a new task."""
        self.current_task = TaskMetrics(
            task_id=task_id,
            task_name=task_name,
            task_type=task_type,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

    def end_task(self, success: bool, final_status: str = "", error: str = "") -> TaskMetrics:
        """End tracking the current task."""
        if self.current_task is None:
            raise ValueError("No task in progress")

        task = self.current_task
        task.end_time = datetime.now(timezone.utc).isoformat()
        task.success = success
        task.final_status = final_status
        task.error_message = error

        # Calculate duration
        start = datetime.fromisoformat(task.start_time)
        end = datetime.fromisoformat(task.end_time)
        task.duration_seconds = (end - start).total_seconds()

        # Collect graph metrics
        try:
            db = get_db()
            if db is not None:
                task.nodes_created = db.node_count
                task.edges_created = db.edge_count
                # Get type distributions using iter_nodes
                for node in db.iter_nodes():
                    # NodeData uses 'type' field, not 'node_type'
                    ntype = node.type.value if hasattr(node.type, 'value') else str(node.type)
                    task.node_types[ntype] = task.node_types.get(ntype, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to collect graph metrics: {e}")

        # Collect LLM metrics from diagnostics
        try:
            diag = get_diagnostics()
            if diag._llm_calls:
                task.llm_calls = len(diag._llm_calls)
                task.total_input_tokens = sum(c.input_tokens for c in diag._llm_calls)
                task.total_output_tokens = sum(c.output_tokens for c in diag._llm_calls)
                task.total_llm_time_ms = sum(c.duration_ms for c in diag._llm_calls)
                for c in diag._llm_calls:
                    task.schemas_used[c.schema_name] = task.schemas_used.get(c.schema_name, 0) + 1
            if diag._phase_metrics:
                task.phases_completed = [p.phase_name for p in diag._phase_metrics]
                task.iterations = len([p for p in diag._phase_metrics if p.phase_name == "BUILD"])
        except Exception as e:
            logger.warning(f"Failed to collect LLM metrics: {e}")

        # Estimate cost (rough: $3/1M input, $15/1M output for Claude)
        task.estimated_cost_usd = (
            (task.total_input_tokens * 3 / 1_000_000) +
            (task.total_output_tokens * 15 / 1_000_000)
        )

        # Add to run metrics
        self.run_metrics.task_metrics.append(task)
        self.run_metrics.tasks_attempted += 1
        if success:
            self.run_metrics.tasks_passed += 1
        else:
            self.run_metrics.tasks_failed += 1
            if error:
                self.run_metrics.failure_reasons.append(f"{task.task_id}: {error}")

        self.current_task = None
        return task

    def finalize(self) -> RunMetrics:
        """Finalize the run and compute aggregates."""
        run = self.run_metrics
        run.end_time = datetime.now(timezone.utc).isoformat()

        start = datetime.fromisoformat(run.start_time)
        end = datetime.fromisoformat(run.end_time)
        run.duration_seconds = (end - start).total_seconds()

        # Compute aggregates
        for task in run.task_metrics:
            run.total_nodes += task.nodes_created
            run.total_edges += task.edges_created
            run.total_llm_calls += task.llm_calls
            run.total_tokens += task.total_input_tokens + task.total_output_tokens
            run.total_cost_usd += task.estimated_cost_usd

        # Find extremes
        if run.task_metrics:
            slowest = max(run.task_metrics, key=lambda t: t.duration_seconds)
            run.slowest_task = f"{slowest.task_id} ({slowest.duration_seconds:.1f}s)"

            most_expensive = max(run.task_metrics, key=lambda t: t.estimated_cost_usd)
            run.most_expensive_task = f"{most_expensive.task_id} (${most_expensive.estimated_cost_usd:.4f})"

        return run

    def save(self) -> Path:
        """Save metrics to file."""
        metrics_file = METRICS_DIR / f"{self.run_id}.json"
        with open(metrics_file, "w") as f:
            # Convert dataclasses to dicts
            data = asdict(self.run_metrics)
            json.dump(data, f, indent=2, default=str)
        return metrics_file

    def print_summary(self) -> None:
        """Print a summary of the run."""
        run = self.run_metrics

        print("\n" + "=" * 70)
        print("BENCHMARK RUN SUMMARY")
        print("=" * 70)
        print(f"Run ID: {run.run_id}")
        print(f"Tier: {run.tier}")
        print(f"Duration: {run.duration_seconds:.1f}s")
        print()
        print(f"Tasks: {run.tasks_passed}/{run.tasks_attempted} passed")
        print(f"Total Nodes: {run.total_nodes}")
        print(f"Total Edges: {run.total_edges}")
        print(f"Total LLM Calls: {run.total_llm_calls}")
        print(f"Total Tokens: {run.total_tokens:,}")
        print(f"Estimated Cost: ${run.total_cost_usd:.4f}")
        print()

        if run.slowest_task:
            print(f"Slowest Task: {run.slowest_task}")
        if run.most_expensive_task:
            print(f"Most Expensive: {run.most_expensive_task}")

        if run.failure_reasons:
            print("\nFailures:")
            for reason in run.failure_reasons:
                print(f"  - {reason}")

        print()
        print("Per-Task Results:")
        print("-" * 70)
        for task in run.task_metrics:
            status = "PASS" if task.success else "FAIL"
            print(f"  [{status}] {task.task_id}: {task.task_name}")
            print(f"        Duration: {task.duration_seconds:.1f}s | "
                  f"Nodes: {task.nodes_created} | "
                  f"LLM Calls: {task.llm_calls} | "
                  f"Cost: ${task.estimated_cost_usd:.4f}")
            if not task.success and task.error_message:
                print(f"        Error: {task.error_message[:60]}...")
        print("=" * 70)


# =============================================================================
# TASK EXECUTION
# =============================================================================

def load_golden_set() -> Dict[str, Any]:
    """Load the golden set configuration."""
    with open(GOLDEN_SET_PATH) as f:
        return yaml.safe_load(f)


def get_tasks_for_tier(golden_set: Dict, tier: str) -> List[Dict]:
    """Get tasks for a specific tier."""
    tier_ids = golden_set.get("tiers", {}).get(tier, [])
    tasks = golden_set.get("tasks", [])
    return [t for t in tasks if t["id"] in tier_ids]


def get_task_by_id(golden_set: Dict, task_id: str) -> Optional[Dict]:
    """Get a specific task by ID."""
    for task in golden_set.get("tasks", []):
        if task["id"] == task_id:
            return task
    return None


def setup_run_workspace(run_id: str) -> Path:
    """Create/reset workspace for a run."""
    run_dir = RUNS_DIR / run_id

    # Remove if exists (overwrite on re-run)
    if run_dir.exists():
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def execute_task(
    task: Dict,
    run_dir: Path,
    interactive: bool = False,
    collector: Optional[MetricsCollector] = None,
) -> Tuple[bool, str]:
    """
    Execute a single benchmark task.

    Returns:
        (success, error_message)
    """
    task_id = task["id"]
    task_name = task["name"]
    task_type = task["type"]
    config = task.get("config", {})

    # Create task-specific subdirectory
    task_dir = run_dir / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Executing: {task_id} - {task_name}")
    print(f"Type: {task_type} | Difficulty: {task.get('difficulty', 'unknown')}")
    print(f"{'='*60}")

    # Start metrics
    if collector:
        collector.start_task(task_id, task_name, task_type)

    # Reset global state for isolation
    set_db(ParagonDB())
    reset_llm()
    reset_diagnostics()

    # Initialize diagnostics
    diag = get_diagnostics()
    diag.set_session(task_id)

    try:
        # Get the task input (requirement)
        requirement = task["input"].strip()

        # Extract requirements from success_criteria (handle list format)
        success_criteria = task.get("success_criteria", [])
        req_list = []
        if isinstance(success_criteria, list):
            for c in success_criteria:
                if isinstance(c, dict) and "contains" in c:
                    req_list.extend(c["contains"])
                elif isinstance(c, str):
                    req_list.append(c)

        # Run through orchestrator
        orchestrator = TDDOrchestrator(enable_checkpointing=False)

        result = orchestrator.run(
            session_id=task_id,
            task_id=task_id,
            spec=requirement,
            requirements=req_list,
            max_iterations=config.get("max_iterations", 5),
        )

        # Check success
        final_status = result.get("final_status", "unknown")
        success = final_status == "passed"

        # Save generated artifacts
        db = get_db()
        if db:
            # Export graph state
            graph_state = {
                "nodes": db.node_count,
                "edges": db.edge_count,
            }
            with open(task_dir / "graph_state.json", "w") as f:
                json.dump(graph_state, f, indent=2)

            # Export any CODE nodes as files
            for node in db.iter_nodes():
                ntype = node.type.value if hasattr(node.type, 'value') else str(node.type)
                if ntype == "CODE":
                    content = node.content
                    data = node.data if isinstance(node.data, dict) else {}
                    filename = data.get("filename", f"{node.id[:8]}.py")
                    with open(task_dir / filename, "w") as f:
                        f.write(content)
                    if collector and collector.current_task:
                        collector.current_task.files_created.append(filename)

        error_msg = "" if success else f"Final status: {final_status}"

        if collector:
            collector.end_task(success, final_status, error_msg)

        return success, error_msg

    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Task {task_id} failed with exception")

        if collector:
            collector.end_task(False, "exception", error_msg)

        return False, error_msg


# =============================================================================
# VISUALIZATION
# =============================================================================

def setup_visualization(run_id: str) -> Optional[Any]:
    """Set up Rerun visualization for the run."""
    if not RERUN_AVAILABLE:
        print("Warning: Rerun not available, skipping visualization")
        return None

    try:
        VIZ_DIR.mkdir(parents=True, exist_ok=True)

        # Create RerunLogger - it handles its own output path based on config
        rerun_logger = create_rerun_logger(session_id=f"benchmark_{run_id}")

        if rerun_logger.recording_path:
            print(f"Visualization will be saved to: {rerun_logger.recording_path}")
        return rerun_logger
    except Exception as e:
        print(f"Warning: Failed to initialize Rerun: {e}")
        return None


# =============================================================================
# MAIN HARNESS
# =============================================================================

def run_benchmark(
    tier: Optional[str] = None,
    task_id: Optional[str] = None,
    interactive: bool = False,
    visualize: bool = True,
) -> RunMetrics:
    """
    Run the benchmark harness.

    Args:
        tier: Run all tasks in a tier (smoke, core, full, stress)
        task_id: Run a specific task by ID
        interactive: Enable interactive mode for Research/Dialectic
        visualize: Enable Rerun visualization

    Returns:
        RunMetrics with all collected data
    """
    # Generate run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load golden set
    golden_set = load_golden_set()

    # Determine tasks to run
    if task_id:
        task = get_task_by_id(golden_set, task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        tasks = [task]
        tier_name = "single"
    elif tier:
        tasks = get_tasks_for_tier(golden_set, tier)
        if not tasks:
            raise ValueError(f"No tasks found for tier: {tier}")
        tier_name = tier
    else:
        raise ValueError("Must specify either --tier or --task")

    print(f"\n{'#'*70}")
    print(f"# PARAGON BENCHMARK HARNESS")
    print(f"# Run ID: {run_id}")
    print(f"# Tier: {tier_name}")
    print(f"# Tasks: {len(tasks)}")
    print(f"# Interactive: {interactive}")
    print(f"# Visualization: {visualize}")
    print(f"{'#'*70}")

    # Setup workspace
    run_dir = setup_run_workspace(run_id)
    print(f"\nWorkspace: {run_dir}")

    # Setup metrics collector
    collector = MetricsCollector(run_id, tier_name)

    # Setup visualization
    viz_logger = None
    if visualize:
        viz_logger = setup_visualization(run_id)

    # Execute tasks
    for task in tasks:
        try:
            success, error = execute_task(
                task=task,
                run_dir=run_dir,
                interactive=interactive,
                collector=collector,
            )
        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            break

    # Finalize
    run_metrics = collector.finalize()
    metrics_file = collector.save()

    # Print summary
    collector.print_summary()
    print(f"\nMetrics saved to: {metrics_file}")

    if viz_logger and VIZ_DIR.exists():
        rrd_files = list(VIZ_DIR.glob("*.rrd"))
        if rrd_files:
            print(f"Visualization saved to: {rrd_files[-1]}")

    return run_metrics


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Paragon Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m benchmarks.harness --tier smoke
    python -m benchmarks.harness --tier core --interactive
    python -m benchmarks.harness --task builder_001
    python -m benchmarks.harness --tier full --no-viz
        """
    )

    parser.add_argument(
        "--tier",
        choices=["smoke", "core", "full", "stress"],
        help="Run all tasks in a tier",
    )
    parser.add_argument(
        "--task",
        help="Run a specific task by ID",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode for user questions",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Disable Rerun visualization",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available tasks and tiers",
    )

    args = parser.parse_args()

    if args.list:
        golden_set = load_golden_set()
        print("\nAvailable Tiers:")
        for tier, task_ids in golden_set.get("tiers", {}).items():
            print(f"  {tier}: {task_ids}")
        print("\nAvailable Tasks:")
        for task in golden_set.get("tasks", []):
            print(f"  {task['id']}: {task['name']} ({task['type']})")
        return

    if not args.tier and not args.task:
        parser.error("Must specify either --tier or --task (or --list)")

    try:
        run_benchmark(
            tier=args.tier,
            task_id=args.task,
            interactive=args.interactive,
            visualize=not args.no_viz,
        )
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
