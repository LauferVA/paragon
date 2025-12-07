"""
Unit tests for domain module - code parsing, test linking, and coverage analysis.

This test suite provides comprehensive coverage for:
- domain/code_parser.py: Tree-sitter based parsing
- domain/test_linker.py: Test discovery and linking
- domain/test_coverage.py: Production coverage calculation

Tests follow the Paragon convention:
- Use msgspec.Struct, not Pydantic
- Test both success and failure paths
- Provide clear documentation for each test
"""
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Set

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus

from domain.code_parser import (
    CodeParser,
    ParsedEntity,
    parse_python_file,
    parse_python_directory,
    parse_typescript_file,
    parse_tsx_file,
    parse_typescript_directory,
    ingest_codebase,
)

from domain.test_linker import (
    classify_test_type,
    extract_test_targets,
    count_test_functions,
    discover_test_files,
    create_test_suite_node,
    build_source_index,
    link_test_to_sources,
    discover_tests,
    ingest_tests,
    get_coverage_by_node,
    get_coverage_summary,
    print_coverage_report,
    TestFileInfo,
)

from domain.test_coverage import (
    is_production_file,
    is_production_node_basic,
    is_production_node_with_db,
    get_module_from_path,
    get_module_from_node_id,
    get_production_dir_from_node,
    calculate_coverage,
    CoverageReport,
    NodeCoverage,
    ModuleCoverage,
    PRODUCTION_DIRS,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""
import os
from pathlib import Path

class Calculator:
    """A simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y

def divide(x, y):
    """Divide x by y."""
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y
'''


@pytest.fixture
def sample_typescript_code():
    """Sample TypeScript code for testing."""
    return '''
import { Component } from 'react';

interface User {
    name: string;
    age: number;
}

type UserId = string | number;

class UserService {
    getUser(id: UserId): User {
        return { name: "John", age: 30 };
    }
}

export function processUser(user: User): void {
    console.log(user.name);
}
'''


@pytest.fixture
def sample_tsx_code():
    """Sample TSX/React code for testing."""
    return '''
import React from 'react';

interface Props {
    title: string;
    count: number;
}

export const Counter: React.FC<Props> = ({ title, count }) => {
    return (
        <div>
            <h1>{title}</h1>
            <p>Count: {count}</p>
        </div>
    );
};

function Button({ onClick }: { onClick: () => void }) {
    return <button onClick={onClick}>Click me</button>;
}
'''


@pytest.fixture
def sample_test_file():
    """Sample test file content."""
    return '''
"""Test for calculator module."""
import pytest
from core.calculator import Calculator, multiply, divide

class TestCalculator:
    def test_add(self):
        calc = Calculator()
        assert calc.add(2, 3) == 5

    def test_subtract(self):
        calc = Calculator()
        assert calc.subtract(5, 3) == 2

def test_multiply():
    assert multiply(3, 4) == 12

def test_divide():
    assert divide(10, 2) == 5

def test_divide_by_zero():
    with pytest.raises(ValueError):
        divide(10, 0)
'''


@pytest.fixture
def temp_python_project():
    """Create a temporary Python project structure."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create source directories
        core_dir = root / "core"
        core_dir.mkdir()

        # Create a sample module
        (core_dir / "calculator.py").write_text('''
"""Calculator module."""

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

def multiply(x, y):
    return x * y
''')

        # Create test directory
        tests_dir = root / "tests" / "unit" / "core"
        tests_dir.mkdir(parents=True)

        (tests_dir / "test_calculator.py").write_text('''
import pytest
from core.calculator import Calculator, multiply

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_multiply():
    assert multiply(3, 4) == 12
''')

        yield root


# =============================================================================
# CODE PARSER TESTS
# =============================================================================

class TestCodeParser:
    """Tests for CodeParser class."""

    def test_parser_init_python(self):
        """Test CodeParser initialization with Python language."""
        parser = CodeParser(language="python")
        assert parser.language == "python"
        assert "classes" in parser._queries
        assert "functions" in parser._queries

    def test_parser_init_typescript(self):
        """Test CodeParser initialization with TypeScript language."""
        parser = CodeParser(language="typescript")
        assert parser.language == "typescript"
        assert "classes" in parser._queries
        assert "interfaces" in parser._queries

    def test_parser_init_tsx(self):
        """Test CodeParser initialization with TSX language."""
        parser = CodeParser(language="tsx")
        assert parser.language == "tsx"
        assert "react_components" in parser._queries

    def test_parser_init_invalid_language(self):
        """Test CodeParser initialization with invalid language raises error."""
        with pytest.raises(ValueError) as exc_info:
            CodeParser(language="invalid")
        assert "not supported" in str(exc_info.value)

    def test_parse_python_content(self, sample_python_code):
        """Test parsing Python code content."""
        parser = CodeParser(language="python")
        nodes, edges = parser.parse_content(
            sample_python_code.encode('utf-8'),
            module_name="calculator",
            file_path="calculator.py"
        )

        # Should have module node + classes + functions + imports
        assert len(nodes) > 0

        # Find module node
        module_nodes = [n for n in nodes if n.type == NodeType.CODE.value and n.data.get("kind") == "module"]
        assert len(module_nodes) == 1
        assert module_nodes[0].data["name"] == "calculator"

        # Find class nodes
        class_nodes = [n for n in nodes if n.type == NodeType.CLASS.value]
        assert len(class_nodes) == 1
        assert class_nodes[0].data["name"] == "Calculator"

        # Find function nodes (top-level functions, not methods)
        func_nodes = [n for n in nodes if n.type == NodeType.FUNCTION.value and not n.data.get("parent")]
        assert len(func_nodes) >= 2  # multiply, divide
        func_names = {n.data["name"] for n in func_nodes}
        assert "multiply" in func_names
        assert "divide" in func_names

        # Find method nodes
        method_nodes = [n for n in nodes if n.type == NodeType.FUNCTION.value and n.data.get("parent")]
        assert len(method_nodes) >= 2  # add, subtract
        method_names = {n.data["name"] for n in method_nodes}
        assert "add" in method_names
        assert "subtract" in method_names

        # Check edges - module should CONTAIN classes and functions
        contains_edges = [e for e in edges if e.type == EdgeType.CONTAINS.value]
        assert len(contains_edges) > 0

    def test_parse_typescript_content(self, sample_typescript_code):
        """Test parsing TypeScript code content."""
        parser = CodeParser(language="typescript")
        nodes, edges = parser.parse_content(
            sample_typescript_code.encode('utf-8'),
            module_name="user_service",
            file_path="user_service.ts"
        )

        # Should have module, class, interface, type alias, function
        assert len(nodes) > 0

        # Find interface nodes
        interface_nodes = [n for n in nodes if n.data.get("kind") == "interface"]
        assert len(interface_nodes) >= 1
        interface_names = {n.data["name"] for n in interface_nodes}
        assert "User" in interface_names

        # Find type alias nodes
        type_nodes = [n for n in nodes if n.data.get("kind") == "type"]
        assert len(type_nodes) >= 1
        type_names = {n.data["name"] for n in type_nodes}
        assert "UserId" in type_names

        # Find class nodes
        class_nodes = [n for n in nodes if n.type == NodeType.CLASS.value]
        assert len(class_nodes) >= 1
        assert class_nodes[0].data["name"] == "UserService"

    def test_parse_tsx_content(self, sample_tsx_code):
        """Test parsing TSX/React code content."""
        parser = CodeParser(language="tsx")
        nodes, edges = parser.parse_content(
            sample_tsx_code.encode('utf-8'),
            module_name="counter",
            file_path="Counter.tsx"
        )

        # Should have React components
        assert len(nodes) > 0

        # Find React component nodes
        component_nodes = [n for n in nodes if n.data.get("kind") == "react_component"]
        assert len(component_nodes) >= 1
        component_names = {n.data["name"] for n in component_nodes}
        assert "Counter" in component_names or "Button" in component_names

    def test_parse_file_not_found(self):
        """Test parse_file with non-existent file raises FileNotFoundError."""
        parser = CodeParser(language="python")
        with pytest.raises(FileNotFoundError):
            parser.parse_file(Path("/nonexistent/file.py"))

    def test_node_text_extraction(self, sample_python_code):
        """Test _node_text method extracts text correctly."""
        parser = CodeParser(language="python")
        content = sample_python_code.encode('utf-8')
        tree = parser.parser.parse(content)

        # Get root node text
        text = parser._node_text(tree.root_node, content)
        assert "Calculator" in text
        assert "multiply" in text

    def test_is_method_detection(self, sample_python_code):
        """Test _is_method correctly identifies methods vs functions."""
        parser = CodeParser(language="python")
        content = sample_python_code.encode('utf-8')
        tree = parser.parser.parse(content)

        # Parse and check that we correctly identify methods
        nodes, _ = parser.parse_content(content, "test_module")

        # Methods should have parent class
        method_nodes = [n for n in nodes if n.type == NodeType.FUNCTION.value and n.data.get("parent")]
        assert len(method_nodes) > 0

        # Functions should not have parent
        func_nodes = [n for n in nodes if n.type == NodeType.FUNCTION.value and not n.data.get("parent")]
        assert len(func_nodes) > 0


class TestParsePythonFile:
    """Tests for parse_python_file convenience function."""

    def test_parse_python_file(self, sample_python_code):
        """Test parse_python_file parses a file successfully."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.py"
            file_path.write_text(sample_python_code)

            nodes, edges = parse_python_file(file_path)

            assert len(nodes) > 0
            assert len(edges) > 0

            # Should have a module node
            module_nodes = [n for n in nodes if n.data.get("kind") == "module"]
            assert len(module_nodes) == 1


class TestParsePythonDirectory:
    """Tests for parse_python_directory function."""

    def test_parse_python_directory_single_file(self, sample_python_code):
        """Test parsing a directory with a single Python file."""
        with TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            (dir_path / "calc.py").write_text(sample_python_code)

            nodes, edges = parse_python_directory(dir_path, recursive=False)

            assert len(nodes) > 0
            assert len(edges) > 0

    def test_parse_python_directory_recursive(self):
        """Test recursive directory parsing."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create nested structure
            (root / "module1.py").write_text("def func1(): pass")
            subdir = root / "subdir"
            subdir.mkdir()
            (subdir / "module2.py").write_text("def func2(): pass")

            # Non-recursive should only find module1
            nodes_nonrecursive, _ = parse_python_directory(root, recursive=False)
            module_names_nonrecursive = {n.data.get("name") for n in nodes_nonrecursive if n.data.get("kind") == "module"}
            assert "module1" in module_names_nonrecursive
            assert "module2" not in module_names_nonrecursive

            # Recursive should find both
            nodes_recursive, _ = parse_python_directory(root, recursive=True)
            module_names_recursive = {n.data.get("name") for n in nodes_recursive if n.data.get("kind") == "module"}
            assert "module1" in module_names_recursive
            assert "module2" in module_names_recursive

    def test_parse_python_directory_excludes_venv(self):
        """Test that exclude patterns work."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create files
            (root / "main.py").write_text("def main(): pass")

            # Create venv directory
            venv_dir = root / ".venv" / "lib"
            venv_dir.mkdir(parents=True)
            (venv_dir / "stdlib.py").write_text("def lib(): pass")

            nodes, _ = parse_python_directory(root, recursive=True)
            module_names = {n.data.get("name") for n in nodes if n.data.get("kind") == "module"}

            # Should find main but not stdlib
            assert "main" in module_names
            assert "stdlib" not in module_names


class TestParseTypeScriptFile:
    """Tests for parse_typescript_file function."""

    def test_parse_typescript_file(self, sample_typescript_code):
        """Test parsing TypeScript file."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "sample.ts"
            file_path.write_text(sample_typescript_code)

            nodes, edges = parse_typescript_file(file_path)

            assert len(nodes) > 0
            interface_nodes = [n for n in nodes if n.data.get("kind") == "interface"]
            assert len(interface_nodes) > 0


class TestParseTsxFile:
    """Tests for parse_tsx_file function."""

    def test_parse_tsx_file(self, sample_tsx_code):
        """Test parsing TSX file."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "Component.tsx"
            file_path.write_text(sample_tsx_code)

            nodes, edges = parse_tsx_file(file_path)

            assert len(nodes) > 0


class TestParseTypeScriptDirectory:
    """Tests for parse_typescript_directory function."""

    def test_parse_typescript_directory(self, sample_typescript_code, sample_tsx_code):
        """Test parsing TypeScript directory with .ts and .tsx files."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "service.ts").write_text(sample_typescript_code)
            (root / "Component.tsx").write_text(sample_tsx_code)

            nodes, edges = parse_typescript_directory(root, recursive=False)

            assert len(nodes) > 0
            module_names = {n.data.get("name") for n in nodes if n.data.get("kind") == "module"}
            assert "service" in module_names or "Component" in module_names

    def test_parse_typescript_directory_excludes_node_modules(self):
        """Test that node_modules is excluded."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "app.ts").write_text("function app() {}")

            # Create node_modules
            nm_dir = root / "node_modules" / "lib"
            nm_dir.mkdir(parents=True)
            (nm_dir / "index.ts").write_text("function lib() {}")

            nodes, _ = parse_typescript_directory(root, recursive=True)
            module_names = {n.data.get("name") for n in nodes if n.data.get("kind") == "module"}

            assert "app" in module_names
            assert "index" not in module_names


class TestIngestCodebase:
    """Tests for ingest_codebase function."""

    def test_ingest_codebase_python(self, temp_python_project):
        """Test ingesting Python codebase into database."""
        db = ParagonDB()

        nodes_added, edges_added = ingest_codebase(
            temp_python_project / "core",
            db,
            recursive=True,
            languages=["python"]
        )

        assert nodes_added > 0
        assert db.node_count == nodes_added

    def test_ingest_codebase_multiple_languages(self):
        """Test ingesting codebase with multiple languages."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            (root / "app.py").write_text("def hello(): pass")
            (root / "service.ts").write_text("function greet() {}")

            db = ParagonDB()
            nodes_added, edges_added = ingest_codebase(
                root,
                db,
                languages=["python", "typescript"]
            )

            assert nodes_added >= 2  # At least 2 module nodes


# =============================================================================
# TEST LINKER TESTS
# =============================================================================

class TestClassifyTestType:
    """Tests for classify_test_type function."""

    def test_classify_unit_test(self):
        """Test classification of unit tests."""
        assert classify_test_type(Path("tests/unit/test_foo.py")) == "unit"

    def test_classify_unit_agents_test(self):
        """Test classification of unit/agents tests."""
        assert classify_test_type(Path("tests/unit/agents/test_tools.py")) == "unit_agents"

    def test_classify_unit_core_test(self):
        """Test classification of unit/core tests."""
        assert classify_test_type(Path("tests/unit/core/test_graph.py")) == "unit_core"

    def test_classify_integration_test(self):
        """Test classification of integration tests."""
        assert classify_test_type(Path("tests/integration/test_api.py")) == "integration"

    def test_classify_e2e_test(self):
        """Test classification of e2e tests."""
        assert classify_test_type(Path("tests/e2e/test_workflow.py")) == "e2e"

    def test_classify_performance_test(self):
        """Test classification of performance tests."""
        assert classify_test_type(Path("tests/performance/test_benchmark.py")) == "performance"

    def test_classify_root_test(self):
        """Test classification of root-level tests."""
        assert classify_test_type(Path("tests/test_main.py")) == "root"

    def test_classify_other_test(self):
        """Test classification of unrecognized tests."""
        assert classify_test_type(Path("somewhere/test_foo.py")) == "other"


class TestExtractTestTargets:
    """Tests for extract_test_targets function."""

    def test_extract_from_imports(self):
        """Test extracting targets from import statements."""
        content = """
from core.calculator import Calculator, add
from domain.parser import CodeParser
import agents.tools
"""
        targets = extract_test_targets(content)

        assert "core" in targets
        assert "core.calculator" in targets
        assert "Calculator" in targets
        assert "add" in targets
        assert "domain" in targets
        assert "domain.parser" in targets
        assert "CodeParser" in targets
        assert "agents" in targets
        assert "agents.tools" in targets

    def test_extract_ignores_stdlib(self):
        """Test that stdlib modules are ignored."""
        content = """
import pytest
import os
from typing import List
"""
        targets = extract_test_targets(content)

        assert "pytest" not in targets
        assert "os" not in targets
        assert "typing" not in targets

    def test_extract_class_references(self):
        """Test extracting class references from instantiation."""
        content = """
def test_foo():
    calc = Calculator()
    parser = CodeParser(language="python")
    result = MyClass(param=42)
"""
        targets = extract_test_targets(content)

        assert "Calculator" in targets
        assert "CodeParser" in targets
        assert "MyClass" in targets

    def test_extract_function_calls(self):
        """Test extracting function call patterns."""
        content = """
def test_processing():
    result = parse_python_file(path)
    data = get_coverage_summary(db)
    build_source_index(db)
"""
        targets = extract_test_targets(content)

        assert "parse_python_file" in targets
        assert "get_coverage_summary" in targets
        assert "build_source_index" in targets


class TestCountTestFunctions:
    """Tests for count_test_functions function."""

    def test_count_test_functions(self, sample_test_file):
        """Test counting test functions in a file."""
        count = count_test_functions(sample_test_file)
        assert count == 5  # 5 test functions

    def test_count_zero_test_functions(self):
        """Test counting when there are no test functions."""
        content = """
def helper_function():
    pass

class NotATest:
    def method(self):
        pass
"""
        count = count_test_functions(content)
        assert count == 0


class TestDiscoverTestFiles:
    """Tests for discover_test_files function."""

    def test_discover_test_files(self, temp_python_project):
        """Test discovering test files in a directory."""
        test_files = discover_test_files(temp_python_project / "tests")

        assert len(test_files) > 0
        assert all(isinstance(tf, TestFileInfo) for tf in test_files)
        assert all(tf.test_count > 0 for tf in test_files)

    def test_discover_excludes_no_tests(self):
        """Test that files without test functions are excluded."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # File with tests
            (root / "test_valid.py").write_text("def test_foo(): pass")

            # File without tests
            (root / "test_empty.py").write_text("def helper(): pass")

            test_files = discover_test_files(root)

            assert len(test_files) == 1
            assert test_files[0].path.name == "test_valid.py"


class TestCreateTestSuiteNode:
    """Tests for create_test_suite_node function."""

    def test_create_test_suite_node(self):
        """Test creating a TEST_SUITE node from TestFileInfo."""
        test_info = TestFileInfo(
            path=Path("tests/unit/test_foo.py"),
            test_count=3,
            test_type="unit",
            targets={"Calculator", "add", "core"},
            content="def test_add(): pass\ndef test_sub(): pass\ndef test_mul(): pass"
        )

        node = create_test_suite_node(test_info)

        assert node.type == NodeType.TEST_SUITE.value
        assert node.status == NodeStatus.VERIFIED.value
        assert node.data["test_count"] == 3
        assert node.data["test_type"] == "unit"
        assert "Calculator" in node.data["targets"]
        assert node.id.startswith("test_suite::")


class TestBuildSourceIndex:
    """Tests for build_source_index function."""

    def test_build_source_index(self):
        """Test building index from graph nodes."""
        db = ParagonDB()

        # Add some source nodes
        module_node = NodeData.create(
            type=NodeType.CODE.value,
            content="",
            data={"kind": "module", "name": "calculator", "path": "core/calculator.py"}
        )
        class_node = NodeData.create(
            type=NodeType.CLASS.value,
            content="",
            data={"name": "Calculator"}
        )
        func_node = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "add"}
        )

        db.add_node(module_node)
        db.add_node(class_node)
        db.add_node(func_node)

        index = build_source_index(db)

        assert "calculator" in index
        assert "Calculator" in index
        assert "add" in index


class TestLinkTestToSources:
    """Tests for link_test_to_sources function."""

    def test_link_test_to_sources(self):
        """Test creating TESTS edges from test to source."""
        db = ParagonDB()

        # Add source node
        calc_node = NodeData.create(
            type=NodeType.CLASS.value,
            content="",
            data={"name": "Calculator"}
        )
        db.add_node(calc_node)

        # Build index
        index = build_source_index(db)

        # Create test info
        test_info = TestFileInfo(
            path=Path("test_calc.py"),
            test_count=2,
            test_type="unit",
            targets={"Calculator", "add"},
            content=""
        )

        test_node = create_test_suite_node(test_info)

        # Link to sources
        edges = link_test_to_sources(test_node, test_info, index)

        assert len(edges) > 0
        assert all(e.type == EdgeType.TESTS.value for e in edges)
        assert any(e.target_id == calc_node.id for e in edges)


class TestDiscoverTests:
    """Tests for discover_tests function."""

    def test_discover_tests(self, temp_python_project):
        """Test discovering tests and creating nodes/edges."""
        db = ParagonDB()

        # First ingest source code
        ingest_codebase(temp_python_project / "core", db, languages=["python"])

        # Now discover tests
        test_nodes, test_edges = discover_tests(temp_python_project / "tests", db)

        assert len(test_nodes) > 0
        assert all(n.type == NodeType.TEST_SUITE.value for n in test_nodes)


class TestIngestTests:
    """Tests for ingest_tests function."""

    def test_ingest_tests(self, temp_python_project):
        """Test ingesting tests into database."""
        db = ParagonDB()

        # Ingest source first
        ingest_codebase(temp_python_project / "core", db, languages=["python"])

        # Ingest tests
        nodes_added, edges_added = ingest_tests(temp_python_project / "tests", db)

        assert nodes_added > 0


class TestGetCoverageByNode:
    """Tests for get_coverage_by_node function."""

    def test_get_coverage_by_node(self):
        """Test getting coverage information by node."""
        db = ParagonDB()

        # Add source node
        func_node = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "calculate", "file": "core/math.py"}
        )
        db.add_node(func_node)

        # Add test node
        test_node = NodeData.create(
            type=NodeType.TEST_SUITE.value,
            content="",
            data={"test_type": "unit", "test_count": 1}
        )
        db.add_node(test_node)

        # Add TESTS edge
        test_edge = EdgeData.create(
            source_id=test_node.id,
            target_id=func_node.id,
            type=EdgeType.TESTS.value,
            metadata={"test_type": "unit"}
        )
        db.add_edge(test_edge)

        # Get coverage
        coverage = get_coverage_by_node(db)

        assert func_node.id in coverage
        assert coverage[func_node.id]["covered"] is True
        assert coverage[func_node.id]["test_count"] > 0


class TestGetCoverageSummary:
    """Tests for get_coverage_summary function."""

    def test_get_coverage_summary(self):
        """Test getting high-level coverage summary."""
        db = ParagonDB()

        # Add 2 functions, 1 covered
        func1 = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "covered_func", "file": "core/mod.py"}
        )
        func2 = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "uncovered_func", "file": "core/mod.py"}
        )
        db.add_node(func1)
        db.add_node(func2)

        # Add test for func1
        test_node = NodeData.create(
            type=NodeType.TEST_SUITE.value,
            content="",
            data={"test_type": "unit"}
        )
        db.add_node(test_node)

        test_edge = EdgeData.create(
            source_id=test_node.id,
            target_id=func1.id,
            type=EdgeType.TESTS.value
        )
        db.add_edge(test_edge)

        # Get summary
        summary = get_coverage_summary(db)

        assert summary["total_nodes"] == 2
        assert summary["covered_nodes"] == 1
        assert summary["coverage_pct"] == 50.0


class TestPrintCoverageReport:
    """Tests for print_coverage_report function."""

    def test_print_coverage_report(self):
        """Test generating coverage report."""
        db = ParagonDB()

        # Add minimal data
        func = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "test_func", "file": "core/mod.py"}
        )
        db.add_node(func)

        # Should not raise
        summary = print_coverage_report(db)
        assert summary is not None


# =============================================================================
# TEST COVERAGE TESTS
# =============================================================================

class TestIsProductionFile:
    """Tests for is_production_file function."""

    def test_production_file_in_core(self):
        """Test that core files are recognized as production."""
        assert is_production_file("core/graph_db.py") is True

    def test_production_file_in_agents(self):
        """Test that agents files are recognized as production."""
        assert is_production_file("agents/orchestrator.py") is True

    def test_test_file_excluded(self):
        """Test that test files are excluded."""
        assert is_production_file("tests/unit/test_foo.py") is False
        assert is_production_file("core/test_utils.py") is False

    def test_benchmark_file_excluded(self):
        """Test that benchmark files are excluded."""
        assert is_production_file("benchmarks/protocol_alpha.py") is False

    def test_venv_excluded(self):
        """Test that venv files are excluded."""
        assert is_production_file(".venv/lib/python/module.py") is False

    def test_empty_path(self):
        """Test that empty path returns False."""
        assert is_production_file("") is False


class TestIsProductionNodeBasic:
    """Tests for is_production_node_basic function."""

    def test_production_function_node(self):
        """Test that function nodes are recognized as production."""
        node = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "calculate", "kind": "function"}
        )
        assert is_production_node_basic(node) is True

    def test_stdlib_node_excluded(self):
        """Test that stdlib nodes are excluded."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="",
            data={"name": "Path", "kind": "import"}
        )
        assert is_production_node_basic(node) is False

    def test_import_node_excluded(self):
        """Test that import nodes are excluded."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="",
            data={"name": "mymodule", "kind": "import"}
        )
        assert is_production_node_basic(node) is False

    def test_empty_init_excluded(self):
        """Test that empty __init__.py modules are excluded."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="",
            data={"name": "__init__", "kind": "module", "lines": 2}
        )
        assert is_production_node_basic(node) is False

    def test_substantial_init_included(self):
        """Test that substantial __init__.py modules are included."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="",
            data={"name": "__init__", "kind": "module", "lines": 50}
        )
        assert is_production_node_basic(node) is True


class TestIsProductionNodeWithDb:
    """Tests for is_production_node_with_db function."""

    def test_production_node_with_valid_dir(self):
        """Test node in production directory."""
        db = ParagonDB()

        # Add module node with correct ID format
        module_node = NodeData(
            id="module::graph_db",
            type=NodeType.CODE.value,
            content="",
            data={"kind": "module", "name": "graph_db", "path": "core/graph_db.py"}
        )
        db.add_node(module_node)

        # Add function node
        func_node = NodeData(
            id="function::graph_db::add_node",
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "add_node"}
        )
        db.add_node(func_node)

        is_prod, prod_dir = is_production_node_with_db(func_node, db)

        assert is_prod is True
        assert prod_dir == "core"


class TestGetModuleFromPath:
    """Tests for get_module_from_path function."""

    def test_get_module_from_path_core(self):
        """Test extracting module from core path."""
        assert get_module_from_path("core/graph_db.py") == "core"

    def test_get_module_from_path_agents(self):
        """Test extracting module from agents path."""
        assert get_module_from_path("agents/orchestrator.py") == "agents"

    def test_get_module_from_path_nested(self):
        """Test extracting module from nested path."""
        assert get_module_from_path("infrastructure/monitoring/logger.py") == "infrastructure"

    def test_get_module_from_path_empty(self):
        """Test empty path returns unknown."""
        assert get_module_from_path("") == "unknown"


class TestGetModuleFromNodeId:
    """Tests for get_module_from_node_id function."""

    def test_get_module_from_function_id(self):
        """Test extracting module from function node ID."""
        node_id = "function::calculator::add"
        assert get_module_from_node_id(node_id) == "calculator"

    def test_get_module_from_class_id(self):
        """Test extracting module from class node ID."""
        node_id = "class::graph_db::ParagonDB"
        assert get_module_from_node_id(node_id) == "graph_db"

    def test_get_module_from_method_id(self):
        """Test extracting module from method node ID."""
        node_id = "method::calculator::Calculator::add"
        assert get_module_from_node_id(node_id) == "calculator"

    def test_get_module_from_invalid_id(self):
        """Test invalid node ID returns unknown."""
        assert get_module_from_node_id("invalid") == "unknown"


class TestGetProductionDirFromNode:
    """Tests for get_production_dir_from_node function."""

    def test_get_production_dir_from_node_data(self):
        """Test getting production dir from node data."""
        db = ParagonDB()

        node = NodeData.create(
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "add_node", "file": "core/graph_db.py"}
        )

        prod_dir = get_production_dir_from_node(node, db)
        assert prod_dir == "core"

    def test_get_production_dir_from_module_lookup(self):
        """Test getting production dir via module node lookup."""
        db = ParagonDB()

        # Add module node
        module_node = NodeData(
            id="module::calculator",
            type=NodeType.CODE.value,
            content="",
            data={"kind": "module", "name": "calculator", "path": "core/calculator.py"}
        )
        db.add_node(module_node)

        # Add function node referencing this module
        func_node = NodeData(
            id="function::calculator::add",
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "add"}
        )
        db.add_node(func_node)

        prod_dir = get_production_dir_from_node(func_node, db)
        assert prod_dir == "core"


class TestCalculateCoverage:
    """Tests for calculate_coverage function."""

    def test_calculate_coverage_empty_db(self):
        """Test coverage calculation on empty database."""
        db = ParagonDB()
        report = calculate_coverage(db)

        assert report.total_nodes == 0
        assert report.covered_nodes == 0
        assert report.coverage_pct == 0.0

    def test_calculate_coverage_with_functions(self):
        """Test coverage calculation with production functions."""
        db = ParagonDB()

        # Add module node
        module_node = NodeData(
            id="module::calculator",
            type=NodeType.CODE.value,
            content="",
            data={"kind": "module", "name": "calculator", "path": "core/calculator.py"}
        )
        db.add_node(module_node)

        # Add production function
        func_node = NodeData(
            id="function::calculator::add",
            type=NodeType.FUNCTION.value,
            content="",
            data={"name": "add", "file": "core/calculator.py"}
        )
        db.add_node(func_node)

        # Add test
        test_node = NodeData.create(
            type=NodeType.TEST_SUITE.value,
            content="",
            data={"test_type": "unit", "test_count": 1}
        )
        db.add_node(test_node)

        test_edge = EdgeData.create(
            source_id=test_node.id,
            target_id=func_node.id,
            type=EdgeType.TESTS.value
        )
        db.add_edge(test_edge)

        report = calculate_coverage(db)

        assert report.total_nodes >= 1
        assert report.covered_nodes >= 1
        assert report.coverage_pct > 0


class TestCoverageReport:
    """Tests for CoverageReport class."""

    def test_coverage_report_properties(self):
        """Test CoverageReport computed properties."""
        report = CoverageReport(
            total_nodes=100,
            covered_nodes=80,
            total_functions=100,
            covered_functions=75
        )

        assert report.coverage_pct == 80.0
        assert report.function_coverage_pct == 75.0

    def test_coverage_report_summary(self):
        """Test summary generation."""
        report = CoverageReport(
            total_nodes=50,
            covered_nodes=25,
            test_suites_found=5,
            test_edges_found=30
        )

        summary = report.summary()
        assert "50" in summary
        assert "25" in summary
        assert "50.0%" in summary

    def test_coverage_report_save(self):
        """Test saving report to file."""
        with TemporaryDirectory() as tmpdir:
            report = CoverageReport(total_nodes=10, covered_nodes=5)
            path = Path(tmpdir) / "report.txt"

            report.save(path)

            assert path.exists()
            content = path.read_text()
            assert "COVERAGE REPORT" in content


class TestNodeCoverage:
    """Tests for NodeCoverage dataclass."""

    def test_node_coverage_creation(self):
        """Test creating NodeCoverage instance."""
        nc = NodeCoverage(
            node_id="func::test",
            name="test_func",
            node_type=NodeType.FUNCTION.value,
            module="core",
            file_path="core/test.py",
            line_number=10,
            is_covered=True,
            test_types={"unit", "integration"},
            test_count=2
        )

        assert nc.name == "test_func"
        assert nc.is_covered is True
        assert len(nc.test_types) == 2


class TestModuleCoverage:
    """Tests for ModuleCoverage dataclass."""

    def test_module_coverage_percentage(self):
        """Test ModuleCoverage percentage calculation."""
        mc = ModuleCoverage(
            name="core",
            total_nodes=100,
            covered_nodes=75
        )

        assert mc.coverage_pct == 75.0

    def test_module_coverage_zero_nodes(self):
        """Test ModuleCoverage with zero nodes."""
        mc = ModuleCoverage(name="empty", total_nodes=0, covered_nodes=0)
        assert mc.coverage_pct == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDomainIntegration:
    """Integration tests for the full domain workflow."""

    def test_full_workflow(self, temp_python_project):
        """Test complete workflow: parse source, discover tests, calculate coverage."""
        db = ParagonDB()

        # Step 1: Ingest source code
        source_nodes, source_edges = ingest_codebase(
            temp_python_project / "core",
            db,
            languages=["python"]
        )
        assert source_nodes > 0

        # Step 2: Ingest tests
        test_nodes, test_edges = ingest_tests(temp_python_project / "tests", db)
        assert test_nodes > 0

        # Step 3: Calculate coverage
        report = calculate_coverage(db)
        assert report.total_nodes > 0

        # Step 4: Get summary
        summary = get_coverage_summary(db)
        assert "total_nodes" in summary
        assert "coverage_pct" in summary
