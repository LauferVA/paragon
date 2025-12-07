"""
PARAGON TEST COVERAGE - Production-Grade Coverage Analysis

Calculates test coverage for production code only, excluding:
- Test files (tests/*, test_*.py, *_test.py)
- Scripts and utilities (scripts/*, benchmarks/*)
- Examples and workspace (examples/*, workspace/*)
- Empty __init__.py files
- Third-party imports (typing, pathlib, etc.)

Coverage is calculated as: nodes with incoming TESTS edges / total production nodes

Usage:
    from domain.test_coverage import calculate_coverage, CoverageReport

    report = calculate_coverage(db)
    print(report.summary())
    report.save(".temp/coverage_report.txt")

    # Or from CLI:
    python -m domain.test_coverage
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from core.graph_db import ParagonDB
from core.schemas import NodeData
from core.ontology import NodeType, EdgeType


# =============================================================================
# CONFIGURATION
# =============================================================================

# Production source directories (relative to project root)
PRODUCTION_DIRS = frozenset({
    "core",
    "agents",
    "infrastructure",
    "api",
    "domain",
    "viz",
    "requirements",
})

# Directories to exclude from coverage calculation
EXCLUDED_DIRS = frozenset({
    "tests",
    "test",
    "scripts",
    "benchmarks",
    "examples",
    "workspace",
    ".temp",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
})

# File patterns to exclude
EXCLUDED_FILE_PATTERNS = [
    r"^test_.*\.py$",      # test_*.py
    r".*_test\.py$",       # *_test.py
    r"^conftest\.py$",     # pytest config
    r"^setup\.py$",        # setup script
    r"^__main__\.py$",     # entry points (optional)
]

# Standard library / third-party modules to exclude from node counting
STDLIB_MODULES = frozenset({
    # typing
    "typing", "Optional", "List", "Dict", "Set", "Tuple", "Any", "Union",
    "Callable", "Awaitable", "TypeVar", "Generic", "Protocol", "Literal",
    "Annotated", "Final", "ClassVar", "Type", "Sequence", "Mapping",
    "Iterable", "Iterator", "Generator", "Coroutine", "AsyncIterator",
    "AsyncGenerator", "NamedTuple", "TypedDict", "overload", "cast",
    # builtins
    "str", "int", "float", "bool", "bytes", "list", "dict", "set", "tuple",
    "None", "True", "False", "Exception", "Error", "Warning", "ValueError",
    "TypeError", "KeyError", "IndexError", "AttributeError", "RuntimeError",
    "NotImplementedError", "StopIteration", "AssertionError",
    # pathlib
    "Path", "PurePath", "PosixPath", "WindowsPath",
    # dataclasses
    "dataclass", "field", "dataclasses",
    # enum
    "Enum", "IntEnum", "StrEnum", "auto", "enum",
    # common stdlib
    "os", "sys", "re", "json", "time", "datetime", "uuid", "hashlib",
    "functools", "itertools", "collections", "contextlib", "copy",
    "io", "tempfile", "shutil", "subprocess", "threading", "asyncio",
    "logging", "warnings", "traceback", "inspect", "abc", "weakref",
    # third-party we don't own
    "msgspec", "rustworkx", "rx", "pytest", "unittest", "mock",
    "sqlite3", "polars", "pandas", "numpy", "requests", "httpx",
    "pydantic", "fastapi", "uvicorn", "granian", "litellm",
    "tree_sitter", "tree_sitter_python", "tree_sitter_typescript",
    "rerun", "langgraph", "langchain",
})

# Node types that represent actual production code requiring test coverage
# Only FUNCTION nodes (functions + methods) - aligns with industry standard coverage
# Classes are covered implicitly via their methods
PRODUCTION_NODE_TYPES = frozenset({
    NodeType.FUNCTION.value,
})


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class NodeCoverage:
    """Coverage information for a single node."""
    node_id: str
    name: str
    node_type: str
    module: str
    file_path: str
    line_number: int
    is_covered: bool
    test_types: Set[str] = field(default_factory=set)
    test_count: int = 0


@dataclass
class ModuleCoverage:
    """Coverage information for a module."""
    name: str
    total_nodes: int = 0
    covered_nodes: int = 0
    functions: int = 0
    functions_covered: int = 0
    classes: int = 0
    classes_covered: int = 0

    @property
    def coverage_pct(self) -> float:
        return (self.covered_nodes / self.total_nodes * 100) if self.total_nodes > 0 else 0.0


@dataclass
class CoverageReport:
    """Complete coverage report."""
    total_nodes: int = 0
    covered_nodes: int = 0
    total_functions: int = 0
    covered_functions: int = 0
    total_classes: int = 0
    covered_classes: int = 0

    by_module: Dict[str, ModuleCoverage] = field(default_factory=dict)
    by_test_type: Dict[str, int] = field(default_factory=dict)
    covered_nodes_list: List[NodeCoverage] = field(default_factory=list)
    uncovered_nodes_list: List[NodeCoverage] = field(default_factory=list)

    # Metadata
    excluded_nodes: int = 0
    test_suites_found: int = 0
    test_edges_found: int = 0

    @property
    def coverage_pct(self) -> float:
        return (self.covered_nodes / self.total_nodes * 100) if self.total_nodes > 0 else 0.0

    @property
    def function_coverage_pct(self) -> float:
        return (self.covered_functions / self.total_functions * 100) if self.total_functions > 0 else 0.0

    @property
    def class_coverage_pct(self) -> float:
        return (self.covered_classes / self.total_classes * 100) if self.total_classes > 0 else 0.0

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "=" * 80,
            "PARAGON TEST COVERAGE REPORT",
            "Production code only (functions/methods)",
            "=" * 80,
            "",
            "SUMMARY",
            "-" * 80,
            f"Total functions:            {self.total_nodes:>6}",
            f"Covered:                    {self.covered_nodes:>6}",
            f"Coverage:                   {self.coverage_pct:>6.1f}%",
            "",
            f"Test suites found:          {self.test_suites_found:>6}",
            f"Test edges (links):         {self.test_edges_found:>6}",
            "",
        ]
        return "\n".join(lines)

    def full_report(self) -> str:
        """Generate full detailed report."""
        lines = [self.summary()]

        # By module
        lines.append("COVERAGE BY MODULE")
        lines.append("-" * 80)
        lines.append(f"{'Module':<20} {'Total':>8} {'Covered':>8} {'Pct':>8} {'Funcs':>8} {'Classes':>8}")
        lines.append("-" * 80)

        for name in sorted(self.by_module.keys()):
            mod = self.by_module[name]
            lines.append(
                f"{name:<20} {mod.total_nodes:>8} {mod.covered_nodes:>8} "
                f"{mod.coverage_pct:>7.1f}% {mod.functions:>8} {mod.classes:>8}"
            )
        lines.append("")

        # By test type
        if self.by_test_type:
            lines.append("COVERAGE BY TEST TYPE")
            lines.append("-" * 80)
            for test_type, count in sorted(self.by_test_type.items(), key=lambda x: -x[1]):
                lines.append(f"  {test_type:<30} {count:>6} nodes covered")
            lines.append("")

        # Covered nodes (sample)
        lines.append("COVERED NODES (sample)")
        lines.append("-" * 80)
        lines.append(f"{'Name':<35} {'Type':<10} {'Module':<15} {'Test Types':<20}")
        lines.append("-" * 80)

        for nc in sorted(self.covered_nodes_list, key=lambda x: (x.module, x.name))[:40]:
            test_types_str = ", ".join(sorted(nc.test_types)[:3])
            if len(nc.test_types) > 3:
                test_types_str += "..."
            lines.append(f"{nc.name:<35} {nc.node_type:<10} {nc.module:<15} {test_types_str:<20}")

        if len(self.covered_nodes_list) > 40:
            lines.append(f"  ... and {len(self.covered_nodes_list) - 40} more")
        lines.append("")

        # Uncovered nodes (sample)
        lines.append("UNCOVERED NODES (sample)")
        lines.append("-" * 80)

        for nc in sorted(self.uncovered_nodes_list, key=lambda x: (x.module, x.name))[:40]:
            lines.append(f"  {nc.module}/{nc.name} ({nc.node_type}) - {nc.file_path}:{nc.line_number}")

        if len(self.uncovered_nodes_list) > 40:
            lines.append(f"  ... and {len(self.uncovered_nodes_list) - 40} more")
        lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def save(self, path: Path | str) -> None:
        """Save report to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.full_report())
        print(f"Coverage report saved to: {path}")


# =============================================================================
# FILTERING FUNCTIONS
# =============================================================================

def is_production_file(file_path: str) -> bool:
    """
    Check if a file path represents production code.

    Returns True if the file is in a production directory and not excluded.
    """
    if not file_path:
        return False

    path = Path(file_path)
    parts = path.parts

    if not parts:
        return False

    # Check if in excluded directory
    for part in parts:
        if part in EXCLUDED_DIRS:
            return False

    # Check if matches excluded file pattern
    filename = path.name
    for pattern in EXCLUDED_FILE_PATTERNS:
        if re.match(pattern, filename):
            return False

    # Check if in production directory
    root_dir = parts[0]
    return root_dir in PRODUCTION_DIRS


def is_production_node_basic(node: NodeData) -> bool:
    """
    Basic check if a node represents production code worth tracking.

    This version doesn't require db access - use is_production_node_with_db
    for full production directory lookup.

    Filters out:
    - Standard library / third-party imports
    - Empty __init__ modules
    - Import nodes
    """
    # Must have data
    if not node.data:
        return False

    name = node.data.get("name", "")

    # Filter out stdlib/third-party
    if name in STDLIB_MODULES:
        return False

    # Filter out private/dunder that are just re-exports
    if name.startswith("__") and name.endswith("__") and name not in {"__init__", "__main__"}:
        return False

    # For import nodes, skip
    kind = node.data.get("kind", "")
    if kind == "import":
        return False

    # For CODE nodes (modules), filter out empty __init__.py
    if node.type == NodeType.CODE.value:
        kind = node.data.get("kind", "")
        if kind == "module" and name == "__init__":
            # Only include if it has actual content
            lines = node.data.get("lines", 0)
            if lines < 5:  # Likely just imports/re-exports
                return False

    return True


def is_production_node(node: NodeData) -> bool:
    """Legacy wrapper - for basic checks without db access."""
    return is_production_node_basic(node)


def is_production_node_with_db(node: NodeData, db: ParagonDB) -> Tuple[bool, Optional[str]]:
    """
    Check if a node represents production code, with full directory lookup.

    Returns:
        Tuple of (is_production, production_dir)
    """
    # Basic checks first
    if not is_production_node_basic(node):
        return False, None

    # Get production directory
    prod_dir = get_production_dir_from_node(node, db)
    if prod_dir is None:
        return False, None

    return True, prod_dir


def get_module_from_path(file_path: str) -> str:
    """Extract module name from file path."""
    if not file_path:
        return "unknown"

    path = Path(file_path)
    parts = path.parts

    if not parts:
        return "unknown"

    # Return first part that's a production dir
    for part in parts:
        if part in PRODUCTION_DIRS:
            return part

    return parts[0] if parts else "unknown"


def get_module_from_node_id(node_id: str) -> str:
    """
    Extract module info from node ID.

    Node ID formats:
    - module::module_name
    - class::module_name::ClassName
    - method::module_name::ClassName::method_name
    - function::module_name::func_name
    - import::module_name::ImportName
    """
    parts = node_id.split("::")
    if len(parts) >= 2:
        # Second part is usually the source module name
        module_name = parts[1]
        return module_name
    return "unknown"


def get_production_dir_from_node(node: NodeData, db: ParagonDB) -> Optional[str]:
    """
    Determine which production directory a node belongs to.

    Checks:
    1. Direct file path in node data
    2. Module node via CONTAINS edge
    3. Node ID parsing
    """
    # Check direct path
    if node.data:
        file_path = node.data.get("file") or node.data.get("path", "")
        if file_path:
            module = get_module_from_path(file_path)
            if module != "unknown":
                return module

    # Try to find parent module via node ID
    module_name = get_module_from_node_id(node.id)

    # Check if this module_name maps to a known production directory
    # by looking up the module node
    module_node_id = f"module::{module_name}"
    module_node = db.get_node(module_node_id)

    if module_node and module_node.data:
        path = module_node.data.get("path", "")
        if path:
            return get_module_from_path(path)

    return None


# =============================================================================
# COVERAGE CALCULATION
# =============================================================================

def calculate_coverage(db: ParagonDB) -> CoverageReport:
    """
    Calculate test coverage for production code.

    Args:
        db: ParagonDB instance with ingested source and tests

    Returns:
        CoverageReport with detailed coverage information
    """
    report = CoverageReport()

    # Count test suites
    test_suites = list(db.get_nodes_by_type(NodeType.TEST_SUITE.value))
    report.test_suites_found = len(test_suites)

    # Collect all production nodes with their production directory
    production_nodes: List[Tuple[NodeData, str]] = []  # (node, prod_dir)
    excluded_count = 0

    for node_type in PRODUCTION_NODE_TYPES:
        for node in db.get_nodes_by_type(node_type):
            is_prod, prod_dir = is_production_node_with_db(node, db)
            if is_prod and prod_dir:
                production_nodes.append((node, prod_dir))
            else:
                excluded_count += 1

    report.excluded_nodes = excluded_count

    # Build module stats
    module_stats: Dict[str, ModuleCoverage] = defaultdict(ModuleCoverage)
    test_type_counts: Dict[str, int] = defaultdict(int)

    # Analyze each production node
    for node, module in production_nodes:
        name = node.data.get("name", node.id) if node.data else node.id
        line_number = node.data.get("start_line", 0) if node.data else 0

        # Get file path - either from node or from parent module
        file_path = node.data.get("file") or node.data.get("path", "") if node.data else ""
        if not file_path:
            # Try to get from module node via node ID
            module_name = get_module_from_node_id(node.id)
            module_node_id = f"module::{module_name}"
            module_node = db.get_node(module_node_id)
            if module_node and module_node.data:
                file_path = module_node.data.get("path", "")

        # Initialize module if needed
        if module not in module_stats:
            module_stats[module] = ModuleCoverage(name=module)

        mod = module_stats[module]
        mod.total_nodes += 1

        if node.type == NodeType.FUNCTION.value:
            mod.functions += 1
            report.total_functions += 1
        elif node.type == NodeType.CLASS.value:
            mod.classes += 1
            report.total_classes += 1

        report.total_nodes += 1

        # Check for TESTS edges
        incoming = db.get_incoming_edges(node.id)
        test_edges = [e for e in incoming if e.get("type") == EdgeType.TESTS.value]

        test_types: Set[str] = set()
        for edge in test_edges:
            report.test_edges_found += 1
            # Get test type from the source TEST_SUITE node
            test_node_id = edge.get("source")
            if test_node_id:
                test_node = db.get_node(test_node_id)
                if test_node and test_node.data:
                    test_type = test_node.data.get("test_type", "unknown")
                    test_types.add(test_type)

        is_covered = len(test_edges) > 0

        node_coverage = NodeCoverage(
            node_id=node.id,
            name=name,
            node_type=node.type,
            module=module,
            file_path=file_path,
            line_number=line_number,
            is_covered=is_covered,
            test_types=test_types,
            test_count=len(test_edges),
        )

        if is_covered:
            report.covered_nodes += 1
            mod.covered_nodes += 1
            report.covered_nodes_list.append(node_coverage)

            if node.type == NodeType.FUNCTION.value:
                mod.functions_covered += 1
                report.covered_functions += 1
            elif node.type == NodeType.CLASS.value:
                mod.classes_covered += 1
                report.covered_classes += 1

            for tt in test_types:
                test_type_counts[tt] += 1
        else:
            report.uncovered_nodes_list.append(node_coverage)

    report.by_module = dict(module_stats)
    report.by_test_type = dict(test_type_counts)

    return report


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run coverage analysis from command line."""
    from domain import ingest_project

    print("=" * 70)
    print("PARAGON PRODUCTION COVERAGE ANALYSIS")
    print("=" * 70)
    print()

    # Initialize and ingest
    print("Phase 1: Ingesting codebase...")
    db = ParagonDB()
    stats = ingest_project(
        Path("."),
        db,
        source_dirs=list(PRODUCTION_DIRS),
    )
    print(f"  Ingested {stats['source_nodes']} source nodes")
    print(f"  Discovered {stats['test_nodes']} test suites")
    print(f"  Created {stats['test_edges']} test links")
    print()

    # Calculate coverage
    print("Phase 2: Calculating production coverage...")
    report = calculate_coverage(db)

    # Print summary
    print(report.summary())

    # Save full report
    output_path = Path(".temp/production_coverage.txt")
    report.save(output_path)

    # Print module breakdown
    print("COVERAGE BY MODULE")
    print("-" * 70)
    for name in sorted(report.by_module.keys()):
        mod = report.by_module[name]
        bar = "█" * int(mod.coverage_pct / 5) + "░" * (20 - int(mod.coverage_pct / 5))
        print(f"  {name:<15} {bar} {mod.coverage_pct:>5.1f}% ({mod.covered_nodes}/{mod.total_nodes})")
    print()

    return report


if __name__ == "__main__":
    main()
