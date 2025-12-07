# Domain layer - structural parsing, code analysis, and test linking
"""
PARAGON DOMAIN LAYER

This module provides:
- CodeParser: Tree-sitter based structural parsing
- TestLinker: Test discovery and source linking
- ingest_project: Unified project ingestion

Usage:
    from domain import ingest_project
    from core.graph_db import ParagonDB

    db = ParagonDB()
    stats = ingest_project(Path("."), db)
    print(f"Ingested {stats['source_nodes']} source nodes")
    print(f"Linked {stats['test_edges']} tests")
"""
from pathlib import Path
from typing import Dict, List, Optional

from .code_parser import (
    CodeParser,
    ParsedEntity,
    ingest_codebase,
    parse_python_directory,
    parse_python_file,
    parse_typescript_directory,
    parse_typescript_file,
    parse_tsx_file,
)
from .test_linker import (
    discover_tests,
    ingest_tests,
    get_coverage_by_node,
    get_coverage_summary,
    print_coverage_report,
    classify_test_type,
    extract_test_targets,
)
from .test_coverage import (
    calculate_coverage,
    CoverageReport,
    NodeCoverage,
    ModuleCoverage,
    is_production_file,
    is_production_node,
    PRODUCTION_DIRS,
)


def ingest_project(
    project_root: Path,
    db,
    source_dirs: Optional[List[str]] = None,
    test_dir: str = "tests",
    languages: Optional[List[str]] = None,
) -> Dict:
    """
    Ingest an entire project (source + tests) into ParagonDB.

    This is the main entry point for project ingestion. It:
    1. Parses source directories into CODE/FUNCTION/CLASS nodes
    2. Discovers test files and creates TEST_SUITE nodes
    3. Links tests to source via TESTS edges

    Args:
        project_root: Root directory of the project
        db: ParagonDB instance
        source_dirs: List of source directories to parse
                     (default: auto-detect from project root)
        test_dir: Directory containing tests (default: "tests")
        languages: Languages to parse (default: ["python"])

    Returns:
        Dict with ingestion statistics:
        {
            "source_nodes": int,
            "source_edges": int,
            "test_nodes": int,
            "test_edges": int,
            "coverage_pct": float,
        }

    Example:
        from domain import ingest_project
        from core.graph_db import ParagonDB

        db = ParagonDB()
        stats = ingest_project(Path("."), db, source_dirs=["core", "agents"])
        print(f"Coverage: {stats['coverage_pct']:.1f}%")
    """
    project_root = Path(project_root)
    languages = languages or ["python"]

    # Auto-detect source directories if not specified
    if source_dirs is None:
        # Common Python project layouts
        candidates = [
            "src", "lib", "core", "agents", "infrastructure",
            "api", "domain", "services", "app", "pkg",
        ]
        source_dirs = [
            d for d in candidates
            if (project_root / d).is_dir()
        ]
        # If no standard dirs found, use project root
        if not source_dirs:
            source_dirs = ["."]

    # Phase 1: Ingest source code
    total_source_nodes = 0
    total_source_edges = 0

    for src_dir in source_dirs:
        src_path = project_root / src_dir
        if not src_path.exists():
            continue

        try:
            nodes, edges = ingest_codebase(src_path, db, languages=languages)
            total_source_nodes += nodes
            total_source_edges += edges
        except Exception as e:
            print(f"Warning: Failed to ingest {src_dir}: {e}")

    # Phase 2: Discover and link tests
    test_path = project_root / test_dir
    test_nodes = 0
    test_edges = 0

    if test_path.exists():
        try:
            test_nodes, test_edges = ingest_tests(test_path, db)
        except Exception as e:
            print(f"Warning: Failed to ingest tests: {e}")

    # Phase 3: Calculate coverage
    summary = get_coverage_summary(db)

    return {
        "source_nodes": total_source_nodes,
        "source_edges": total_source_edges,
        "test_nodes": test_nodes,
        "test_edges": test_edges,
        "coverage_pct": summary["coverage_pct"],
        "coverage_summary": summary,
    }


__all__ = [
    # Code parser
    "CodeParser",
    "ParsedEntity",
    "ingest_codebase",
    "parse_python_directory",
    "parse_python_file",
    "parse_typescript_directory",
    "parse_typescript_file",
    "parse_tsx_file",
    # Test linker
    "discover_tests",
    "ingest_tests",
    "get_coverage_by_node",
    "get_coverage_summary",
    "print_coverage_report",
    "classify_test_type",
    "extract_test_targets",
    # Production coverage
    "calculate_coverage",
    "CoverageReport",
    "NodeCoverage",
    "ModuleCoverage",
    "is_production_file",
    "is_production_node",
    "PRODUCTION_DIRS",
    # Unified ingestion
    "ingest_project",
]
