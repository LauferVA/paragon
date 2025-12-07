"""
PARAGON TEST LINKER - Test Discovery and Source Linking

Discovers test files, creates TEST_SUITE nodes, and links them to source CODE nodes
via TESTS edges. This enables graph-based coverage analysis.

Architecture:
    1. Test Discovery: Finds test files matching patterns (test_*.py, *_test.py)
    2. Target Extraction: Parses imports and function calls to identify what is tested
    3. Node Creation: Creates TEST_SUITE nodes with metadata
    4. Edge Creation: Links TEST_SUITE -> CODE via TESTS edges

Usage:
    from domain.test_linker import discover_tests, link_tests_to_source

    # After ingesting source code:
    db = ParagonDB()
    ingest_codebase(Path("src"), db)

    # Discover and link tests:
    test_nodes, test_edges = discover_tests(Path("tests"), db)
    db.add_nodes_batch(test_nodes)
    db.add_edges_batch(test_edges)

    # Or use the unified function:
    from domain.test_linker import ingest_tests
    ingest_tests(Path("tests"), db)
"""
import re
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TestFileInfo:
    """Information about a discovered test file."""
    path: Path
    test_count: int
    test_type: str  # unit, integration, e2e, performance, security, etc.
    targets: Set[str]  # Names of modules/classes/functions being tested
    content: str


@dataclass
class TestTarget:
    """A potential target that a test file is testing."""
    name: str
    target_type: str  # "class", "function", "module"
    confidence: float  # 0.0 to 1.0


# =============================================================================
# TEST TYPE CLASSIFICATION
# =============================================================================

def classify_test_type(file_path: Path) -> str:
    """
    Classify a test file by type based on its path.

    Convention:
        tests/unit/         -> unit
        tests/integration/  -> integration
        tests/e2e/          -> e2e
        tests/performance/  -> performance
        tests/security/     -> security
        tests/test_*.py     -> root (uncategorized unit tests)
    """
    path_str = str(file_path)

    # Check for subdirectory patterns
    if '/unit/agents' in path_str:
        return 'unit_agents'
    elif '/unit/core' in path_str:
        return 'unit_core'
    elif '/unit/infrastructure' in path_str:
        return 'unit_infrastructure'
    elif '/unit/' in path_str:
        return 'unit'
    elif '/integration/' in path_str:
        return 'integration'
    elif '/e2e/' in path_str:
        return 'e2e'
    elif '/performance/' in path_str:
        return 'performance'
    elif '/security/' in path_str:
        return 'security'
    elif 'tests/test_' in path_str or path_str.startswith('test_'):
        return 'root'
    else:
        return 'other'


# =============================================================================
# TARGET EXTRACTION
# =============================================================================

# Patterns for extracting test targets from imports
IMPORT_PATTERNS = [
    r'from\s+([\w.]+)\s+import\s+(.+)',  # from module import X, Y
    r'import\s+([\w.]+)',                 # import module
]

# Standard library / common third-party modules to ignore
IGNORED_MODULES = {
    'pytest', 'unittest', 'sys', 'os', 'pathlib', 'typing', 'datetime',
    'tempfile', 'shutil', 're', 'json', 'msgspec', 'uuid', 'io', 'collections',
    'functools', 'itertools', 'contextlib', 'copy', 'time', 'random',
    'asyncio', 'concurrent', 'threading', 'multiprocessing', 'mock',
    'unittest.mock', 'pytest_mock', 'hypothesis', 'faker',
}


def extract_test_targets(content: str) -> Set[str]:
    """
    Extract what a test file is testing based on imports and references.

    Returns a set of target names (modules, classes, functions).
    """
    targets = set()

    # Extract from import statements
    for line in content.split('\n'):
        line = line.strip()

        # from X import Y, Z
        match = re.match(r'from\s+([\w.]+)\s+import\s+(.+)', line)
        if match:
            module = match.group(1)
            imports = match.group(2)

            # Get base module
            base_module = module.split('.')[0]
            if base_module not in IGNORED_MODULES:
                targets.add(base_module)
                # Also add full module path for precision
                targets.add(module)

            # Extract individual imports
            for imp in imports.split(','):
                imp = imp.strip()
                # Handle "X as Y" aliases
                if ' as ' in imp:
                    imp = imp.split(' as ')[0].strip()
                # Handle parenthetical continuation
                imp = imp.strip('()')
                if imp and not imp.startswith('#'):
                    targets.add(imp)

        # import X
        match = re.match(r'^import\s+([\w.]+)', line)
        if match:
            module = match.group(1)
            base_module = module.split('.')[0]
            if base_module not in IGNORED_MODULES:
                targets.add(base_module)
                targets.add(module)

    # Extract class references (PascalCase identifiers followed by parenthesis)
    class_refs = re.findall(r'\b([A-Z][a-zA-Z0-9]+)\s*\(', content)
    for ref in class_refs:
        if ref not in {'True', 'False', 'None', 'Exception', 'Error', 'Warning'}:
            targets.add(ref)

    # Extract function call patterns (common test patterns)
    func_patterns = [
        r'\b(get_\w+)\s*\(',
        r'\b(add_\w+)\s*\(',
        r'\b(create_\w+)\s*\(',
        r'\b(parse_\w+)\s*\(',
        r'\b(check_\w+)\s*\(',
        r'\b(validate_\w+)\s*\(',
        r'\b(build_\w+)\s*\(',
        r'\b(load_\w+)\s*\(',
        r'\b(save_\w+)\s*\(',
        r'\b(process_\w+)\s*\(',
        r'\b(compute_\w+)\s*\(',
        r'\b(calculate_\w+)\s*\(',
    ]
    for pattern in func_patterns:
        for match in re.findall(pattern, content):
            targets.add(match)

    return targets


def count_test_functions(content: str) -> int:
    """Count the number of test functions in a file."""
    return len(re.findall(r'^\s*def test_', content, re.MULTILINE))


# =============================================================================
# TEST DISCOVERY
# =============================================================================

def discover_test_files(
    test_dir: Path,
    patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> List[TestFileInfo]:
    """
    Discover all test files in a directory.

    Args:
        test_dir: Root directory to search for tests
        patterns: Glob patterns for test files (default: ["test_*.py", "*_test.py"])
        exclude_patterns: Patterns to exclude (default: ["__pycache__", ".venv"])

    Returns:
        List of TestFileInfo for each discovered test file
    """
    test_dir = Path(test_dir)
    if not test_dir.exists():
        return []

    patterns = patterns or ["**/test_*.py", "**/*_test.py"]
    exclude_patterns = exclude_patterns or ["__pycache__", ".venv", "venv", ".git"]

    test_files = []
    seen_paths = set()  # Deduplicate

    for pattern in patterns:
        for test_path in test_dir.glob(pattern):
            # Skip excluded patterns
            if any(excl in str(test_path) for excl in exclude_patterns):
                continue

            # Deduplicate
            if test_path in seen_paths:
                continue
            seen_paths.add(test_path)

            try:
                content = test_path.read_text()
                test_count = count_test_functions(content)

                # Skip files with no test functions
                if test_count == 0:
                    continue

                test_files.append(TestFileInfo(
                    path=test_path,
                    test_count=test_count,
                    test_type=classify_test_type(test_path),
                    targets=extract_test_targets(content),
                    content=content,
                ))
            except Exception:
                # Skip files we can't read
                continue

    return test_files


# =============================================================================
# TEST NODE CREATION
# =============================================================================

def create_test_suite_node(test_info: TestFileInfo) -> NodeData:
    """
    Create a TEST_SUITE node from test file information.

    The node stores:
    - File path
    - Number of test functions
    - Test type classification
    - First few lines as content summary
    """
    # Create content summary (first ~20 lines)
    lines = test_info.content.split('\n')[:20]
    content_summary = '\n'.join(lines)
    if len(test_info.content.split('\n')) > 20:
        content_summary += f"\n... ({len(lines)} more lines)"

    # Generate stable ID from file path
    node_id = f"test_suite::{test_info.path}"

    return NodeData(
        id=node_id,
        type=NodeType.TEST_SUITE.value,
        content=content_summary,
        status=NodeStatus.VERIFIED.value,
        data={
            "file": str(test_info.path),
            "test_count": test_info.test_count,
            "test_type": test_info.test_type,
            "targets": list(test_info.targets),
        },
    )


# =============================================================================
# TEST LINKING
# =============================================================================

def build_source_index(db) -> Dict[str, str]:
    """
    Build an index of source names to node IDs from the graph.

    Returns a dict mapping names (class names, function names, module names)
    to their node IDs in the graph.
    """
    index = {}

    # Index CODE nodes
    for node in db.get_nodes_by_type(NodeType.CODE.value):
        if node.data:
            name = node.data.get('name')
            if name:
                index[name] = node.id
                # Also index by module.name for qualified lookups
                module = node.data.get('kind')
                if module == 'module':
                    # Module nodes: index by path components
                    path = node.data.get('path', '')
                    if path:
                        parts = Path(path).parts
                        for i in range(len(parts)):
                            partial = '.'.join(parts[i:]).replace('.py', '')
                            index[partial] = node.id

    # Index FUNCTION nodes
    for node in db.get_nodes_by_type(NodeType.FUNCTION.value):
        if node.data:
            name = node.data.get('name')
            if name:
                index[name] = node.id

    # Index CLASS nodes
    for node in db.get_nodes_by_type(NodeType.CLASS.value):
        if node.data:
            name = node.data.get('name')
            if name:
                index[name] = node.id

    return index


def link_test_to_sources(
    test_node: NodeData,
    test_info: TestFileInfo,
    source_index: Dict[str, str],
) -> List[EdgeData]:
    """
    Create TESTS edges from a test node to source nodes.

    Uses the extracted targets and source index to find matches.
    """
    edges = []
    linked_targets = set()  # Avoid duplicate edges

    for target in test_info.targets:
        if target in source_index:
            target_id = source_index[target]
            if target_id not in linked_targets:
                linked_targets.add(target_id)
                edges.append(EdgeData(
                    source_id=test_node.id,
                    target_id=target_id,
                    type=EdgeType.TESTS.value,
                    metadata={
                        "test_type": test_info.test_type,
                        "test_file": str(test_info.path),
                    },
                ))

    return edges


# =============================================================================
# MAIN API
# =============================================================================

def discover_tests(
    test_dir: Path,
    db,
) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Discover tests and create nodes/edges for graph ingestion.

    Args:
        test_dir: Directory containing test files
        db: ParagonDB instance (used to build source index)

    Returns:
        Tuple of (test_nodes, test_edges)
    """
    # Build index of source code
    source_index = build_source_index(db)

    # Discover test files
    test_files = discover_test_files(test_dir)

    nodes = []
    edges = []

    for test_info in test_files:
        # Create test node
        test_node = create_test_suite_node(test_info)
        nodes.append(test_node)

        # Create edges to source
        test_edges = link_test_to_sources(test_node, test_info, source_index)
        edges.extend(test_edges)

    return nodes, edges


def ingest_tests(test_dir: Path, db) -> Tuple[int, int]:
    """
    Discover and ingest tests into the graph.

    Args:
        test_dir: Directory containing test files
        db: ParagonDB instance

    Returns:
        Tuple of (nodes_added, edges_added)
    """
    nodes, edges = discover_tests(test_dir, db)

    # Add nodes
    for node in nodes:
        try:
            db.add_node(node)
        except Exception:
            pass  # Skip duplicates

    # Add edges (only where both endpoints exist)
    valid_edges = []
    for edge in edges:
        try:
            db.add_edge(edge)
            valid_edges.append(edge)
        except Exception:
            pass  # Skip invalid edges

    return len(nodes), len(valid_edges)


# =============================================================================
# COVERAGE ANALYSIS (Graph-Native)
# =============================================================================

def get_coverage_by_node(db) -> Dict[str, Dict]:
    """
    Calculate test coverage for each source node using graph topology.

    Returns a dict mapping node_id to coverage info:
    {
        "node_id": {
            "name": "function_name",
            "type": "FUNCTION",
            "module": "core",
            "test_types": ["unit_core", "integration"],
            "test_count": 3,
            "covered": True,
        }
    }
    """
    coverage = {}

    # Get all source nodes
    source_types = [NodeType.CODE.value, NodeType.FUNCTION.value, NodeType.CLASS.value]

    for node_type in source_types:
        for node in db.get_nodes_by_type(node_type):
            # Get incoming TESTS edges
            incoming = db.get_incoming_edges(node.id)
            test_edges = [e for e in incoming if e.get('type') == EdgeType.TESTS.value]

            # Extract test types from edges
            test_types = set()
            for edge in test_edges:
                metadata = edge.get('metadata', {})
                if isinstance(metadata, dict):
                    test_type = metadata.get('test_type', 'unknown')
                    test_types.add(test_type)

            # Get node metadata
            name = node.data.get('name', node.id) if node.data else node.id
            module = 'unknown'
            if node.data:
                file_path = node.data.get('file') or node.data.get('path', '')
                if file_path:
                    parts = Path(file_path).parts
                    if parts:
                        module = parts[0]

            coverage[node.id] = {
                "name": name,
                "type": node_type,
                "module": module,
                "test_types": sorted(test_types),
                "test_count": len(test_edges),
                "covered": len(test_edges) > 0,
            }

    return coverage


def get_coverage_summary(db) -> Dict:
    """
    Get high-level coverage statistics.

    Returns:
    {
        "total_nodes": 100,
        "covered_nodes": 30,
        "coverage_pct": 30.0,
        "by_module": {...},
        "by_type": {...},
    }
    """
    coverage = get_coverage_by_node(db)

    total = len(coverage)
    covered = sum(1 for c in coverage.values() if c["covered"])

    # Aggregate by module
    by_module = {}
    for info in coverage.values():
        module = info["module"]
        if module not in by_module:
            by_module[module] = {"total": 0, "covered": 0}
        by_module[module]["total"] += 1
        if info["covered"]:
            by_module[module]["covered"] += 1

    # Calculate percentages
    for module_stats in by_module.values():
        t, c = module_stats["total"], module_stats["covered"]
        module_stats["pct"] = (c / t * 100) if t > 0 else 0.0

    # Aggregate by node type
    by_type = {}
    for info in coverage.values():
        node_type = info["type"]
        if node_type not in by_type:
            by_type[node_type] = {"total": 0, "covered": 0}
        by_type[node_type]["total"] += 1
        if info["covered"]:
            by_type[node_type]["covered"] += 1

    for type_stats in by_type.values():
        t, c = type_stats["total"], type_stats["covered"]
        type_stats["pct"] = (c / t * 100) if t > 0 else 0.0

    return {
        "total_nodes": total,
        "covered_nodes": covered,
        "coverage_pct": (covered / total * 100) if total > 0 else 0.0,
        "by_module": by_module,
        "by_type": by_type,
    }


def print_coverage_report(db, output_path: Optional[Path] = None):
    """
    Generate and optionally write a coverage report.

    Args:
        db: ParagonDB instance with ingested source and tests
        output_path: Optional path to write report (prints to stdout if None)
    """
    coverage = get_coverage_by_node(db)
    summary = get_coverage_summary(db)

    lines = []
    lines.append("=" * 90)
    lines.append("PARAGON COVERAGE REPORT")
    lines.append("Generated by domain/test_linker.py")
    lines.append("=" * 90)
    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append(f"Total source nodes:  {summary['total_nodes']}")
    lines.append(f"Covered nodes:       {summary['covered_nodes']}")
    lines.append(f"Coverage:            {summary['coverage_pct']:.1f}%")
    lines.append("")

    # By module
    lines.append("COVERAGE BY MODULE")
    lines.append("-" * 90)
    lines.append(f"{'Module':<20} {'Total':>8} {'Covered':>8} {'Pct':>10}")
    lines.append("-" * 90)
    for module, stats in sorted(summary["by_module"].items()):
        lines.append(f"{module:<20} {stats['total']:>8} {stats['covered']:>8} {stats['pct']:>9.1f}%")
    lines.append("")

    # By type
    lines.append("COVERAGE BY NODE TYPE")
    lines.append("-" * 90)
    lines.append(f"{'Type':<20} {'Total':>8} {'Covered':>8} {'Pct':>10}")
    lines.append("-" * 90)
    for node_type, stats in sorted(summary["by_type"].items()):
        lines.append(f"{node_type:<20} {stats['total']:>8} {stats['covered']:>8} {stats['pct']:>9.1f}%")
    lines.append("")

    # Covered nodes (sample)
    lines.append("COVERED NODES (showing test types)")
    lines.append("-" * 90)
    lines.append(f"{'Name':<40} {'Type':<12} {'Test Types':<30}")
    lines.append("-" * 90)

    covered_nodes = [(k, v) for k, v in coverage.items() if v["covered"]]
    for node_id, info in sorted(covered_nodes, key=lambda x: x[1]["name"])[:50]:
        test_types_str = ", ".join(info["test_types"])
        lines.append(f"{info['name']:<40} {info['type']:<12} {test_types_str:<30}")

    if len(covered_nodes) > 50:
        lines.append(f"  ... and {len(covered_nodes) - 50} more")
    lines.append("")

    # Uncovered nodes (sample)
    lines.append("UNCOVERED NODES")
    lines.append("-" * 90)
    uncovered = [(k, v) for k, v in coverage.items() if not v["covered"]]
    for node_id, info in sorted(uncovered, key=lambda x: x[1]["name"])[:50]:
        lines.append(f"  - {info['module']}/{info['name']} ({info['type']})")

    if len(uncovered) > 50:
        lines.append(f"  ... and {len(uncovered) - 50} more")

    lines.append("")
    lines.append("=" * 90)

    report = '\n'.join(lines)

    if output_path:
        output_path.write_text(report)
        print(f"Report written to: {output_path}")
    else:
        print(report)

    return summary
