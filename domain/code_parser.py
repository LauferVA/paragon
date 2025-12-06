"""
PARAGON CODE PARSER - The Perception System

Structural code parsing using tree-sitter for fast, accurate AST extraction.

Why tree-sitter over Python's ast module:
1. INCREMENTAL: Can re-parse only changed portions (vital for large codebases)
2. FAULT-TOLERANT: Produces partial ASTs even for broken code
3. MULTI-LANGUAGE: Same API for Python, JS, Rust, Go, etc.
4. FAST: Written in C, exposed via Rust bindings

Architecture:
- CodeParser: Main entry point for parsing source files
- S-Expression queries extract classes, functions, imports
- Output is NodeData/EdgeData ready for graph ingestion

Supported Languages:
- Python (.py)
- TypeScript (.ts)
- TSX/React (.tsx)

Query Pattern:
    tree-sitter uses S-expressions to match AST patterns.
    Example: (function_definition name: (identifier) @func_name)
    This captures the function name into @func_name for extraction.

API Note (tree-sitter 0.25.x):
    Query execution uses QueryCursor with captures() or matches() methods.
    cursor = QueryCursor(query)
    captures = cursor.captures(tree.root_node)  # Returns dict[str, list[Node]]
"""
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Parser, Query, QueryCursor, Tree, Node
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from dataclasses import dataclass

from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# LANGUAGE SETUP
# =============================================================================

# Initialize languages
PY_LANGUAGE = Language(tspython.language())
TS_LANGUAGE = Language(tstypescript.language_typescript())
TSX_LANGUAGE = Language(tstypescript.language_tsx())

# Language registry for file extension mapping
LANGUAGE_REGISTRY = {
    ".py": ("python", PY_LANGUAGE),
    ".ts": ("typescript", TS_LANGUAGE),
    ".tsx": ("tsx", TSX_LANGUAGE),
    ".jsx": ("tsx", TSX_LANGUAGE),  # JSX uses TSX parser
}

# Pre-compiled queries for Python (S-expressions)
# These are the patterns we extract from source code

PYTHON_QUERIES = {
    "classes": """
        (class_definition
            name: (identifier) @class_name
            body: (block) @class_body
        ) @class_def
    """,

    "functions": """
        (function_definition
            name: (identifier) @func_name
            parameters: (parameters) @func_params
            body: (block) @func_body
        ) @func_def
    """,

    "methods": """
        (class_definition
            name: (identifier) @class_name
            body: (block
                (function_definition
                    name: (identifier) @method_name
                ) @method_def
            )
        )
    """,

    "imports": """
        [
            (import_statement
                name: (dotted_name) @import_name
            )
            (import_from_statement
                module_name: (dotted_name) @module_name
                name: (dotted_name) @import_name
            )
            (import_from_statement
                module_name: (dotted_name) @module_name
                name: (aliased_import
                    name: (dotted_name) @import_name
                )
            )
        ]
    """,

    "calls": """
        (call
            function: [
                (identifier) @call_name
                (attribute
                    object: (_) @call_object
                    attribute: (identifier) @call_attr
                )
            ]
        ) @call_expr
    """,
}


# TypeScript/TSX queries
TYPESCRIPT_QUERIES = {
    "classes": """
        (class_declaration
            name: (type_identifier) @class_name
            body: (class_body) @class_body
        ) @class_def
    """,

    "functions": """
        [
            (function_declaration
                name: (identifier) @func_name
                parameters: (formal_parameters) @func_params
                body: (statement_block) @func_body
            ) @func_def
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @func_name
                    value: (arrow_function
                        parameters: (formal_parameters) @func_params
                        body: (_) @func_body
                    )
                )
            ) @func_def
        ]
    """,

    "methods": """
        (class_declaration
            name: (type_identifier) @class_name
            body: (class_body
                (method_definition
                    name: (property_identifier) @method_name
                ) @method_def
            )
        )
    """,

    "imports": """
        [
            (import_statement
                source: (string) @import_source
            )
            (import_statement
                (import_clause
                    (named_imports
                        (import_specifier
                            name: (identifier) @import_name
                        )
                    )
                )
                source: (string) @import_source
            )
            (import_statement
                (import_clause
                    (identifier) @import_name
                )
                source: (string) @import_source
            )
        ]
    """,

    "interfaces": """
        (interface_declaration
            name: (type_identifier) @interface_name
            body: (interface_body) @interface_body
        ) @interface_def
    """,

    "types": """
        (type_alias_declaration
            name: (type_identifier) @type_name
            value: (_) @type_value
        ) @type_def
    """,

    "react_components": """
        [
            (function_declaration
                name: (identifier) @component_name
                return_type: (type_annotation
                    (generic_type
                        name: (type_identifier) @return_type
                    )
                )?
                body: (statement_block) @component_body
            ) @component_def
            (lexical_declaration
                (variable_declarator
                    name: (identifier) @component_name
                    type: (type_annotation)?
                    value: (arrow_function) @component_body
                )
            ) @component_def
        ]
    """,

    "jsx_elements": """
        (jsx_element
            open_tag: (jsx_opening_element
                name: (_) @jsx_tag_name
            )
        ) @jsx_element
    """,

    "exports": """
        [
            (export_statement
                declaration: (_) @export_decl
            ) @export_def
            (export_statement
                (export_clause
                    (export_specifier
                        name: (identifier) @export_name
                    )
                )
            ) @export_def
        ]
    """,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ParsedEntity:
    """Intermediate representation of a parsed code entity."""
    name: str
    entity_type: str  # "class", "function", "method", "import", "call"
    start_line: int
    end_line: int
    start_col: int
    end_col: int
    parent: Optional[str] = None  # For methods, the containing class
    content: Optional[str] = None  # The actual source code
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# =============================================================================
# CODE PARSER
# =============================================================================

class CodeParser:
    """
    Tree-sitter based code parser for structural analysis.

    Extracts:
    - Class definitions
    - Function/method definitions
    - Import statements
    - Function calls (for call graph)
    - TypeScript interfaces and types
    - React/JSX components

    Supported Languages:
    - python (.py)
    - typescript (.ts)
    - tsx (.tsx, .jsx)

    Usage:
        parser = CodeParser()  # Default: Python

        # Parse a file (auto-detects language from extension)
        nodes, edges = parser.parse_file(Path("module.py"))
        nodes, edges = parser.parse_file(Path("Component.tsx"))

        # Or specify language explicitly
        parser = CodeParser(language="typescript")
        nodes, edges = parser.parse_content(source_bytes, "module")

        # Add to graph
        db.add_nodes_batch(nodes)
        db.add_edges_batch(edges)
    """

    SUPPORTED_LANGUAGES = {"python", "typescript", "tsx"}

    def __init__(self, language: str = "python"):
        """
        Initialize parser for specified language.

        Args:
            language: Programming language ("python", "typescript", or "tsx")
        """
        if language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Use one of: {', '.join(sorted(self.SUPPORTED_LANGUAGES))}"
            )

        self.language = language

        # Select language and queries based on type
        if language == "python":
            self._lang = PY_LANGUAGE
            self._query_defs = PYTHON_QUERIES
        elif language == "typescript":
            self._lang = TS_LANGUAGE
            self._query_defs = TYPESCRIPT_QUERIES
        elif language == "tsx":
            self._lang = TSX_LANGUAGE
            self._query_defs = TYPESCRIPT_QUERIES  # TSX uses same queries

        self.parser = Parser(self._lang)

        # Pre-compile queries using new Query constructor (tree-sitter 0.25+)
        self._queries = {}
        for name, query_str in self._query_defs.items():
            try:
                self._queries[name] = Query(self._lang, query_str)
            except Exception as e:
                # Some queries may not be valid for all languages
                pass  # Skip invalid queries silently

    def parse_file(self, path: Path) -> Tuple[List[NodeData], List[EdgeData]]:
        """
        Parse a source file and extract structural elements.

        Args:
            path: Path to source file

        Returns:
            Tuple of (nodes, edges) ready for graph ingestion
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        content = path.read_bytes()
        module_name = path.stem  # filename without extension

        return self.parse_content(content, module_name, str(path))

    def parse_content(
        self,
        content: bytes,
        module_name: str,
        file_path: Optional[str] = None,
    ) -> Tuple[List[NodeData], List[EdgeData]]:
        """
        Parse source content and extract structural elements.

        Args:
            content: Source code as bytes
            module_name: Name for the module node
            file_path: Optional file path for metadata

        Returns:
            Tuple of (nodes, edges)
        """
        # Parse into AST
        tree = self.parser.parse(content)

        # Extract entities
        entities = self._extract_entities(tree, content)

        # Convert to graph nodes/edges
        return self._entities_to_graph(entities, module_name, file_path, content)

    def _extract_entities(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract all code entities from the AST."""
        entities = []

        # Extract classes
        if "classes" in self._queries:
            entities.extend(self._extract_classes(tree, content))

        # Extract top-level functions (not methods)
        if "functions" in self._queries:
            entities.extend(self._extract_functions(tree, content))

        # Extract imports
        if "imports" in self._queries:
            entities.extend(self._extract_imports(tree, content))

        # TypeScript/TSX specific extractions
        if self.language in ("typescript", "tsx"):
            if "interfaces" in self._queries:
                entities.extend(self._extract_interfaces(tree, content))
            if "types" in self._queries:
                entities.extend(self._extract_types(tree, content))
            if "react_components" in self._queries and self.language == "tsx":
                entities.extend(self._extract_react_components(tree, content))

        return entities

    def _extract_classes(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract class definitions with their methods."""
        entities = []
        query = self._queries["classes"]

        # tree-sitter 0.25+ API: use QueryCursor with captures()
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        # captures is dict: {"class_name": [nodes], "class_def": [nodes], ...}
        class_names = captures.get("class_name", [])
        class_defs = captures.get("class_def", [])

        # Match class names with their definitions by position
        for class_def_node in class_defs:
            # Find the class name node within this class definition
            class_name = None
            for name_node in class_names:
                # Check if name_node is within class_def_node
                if (name_node.start_byte >= class_def_node.start_byte and
                    name_node.end_byte <= class_def_node.end_byte):
                    class_name = self._node_text(name_node, content)
                    break

            if class_name:
                entities.append(ParsedEntity(
                    name=class_name,
                    entity_type="class",
                    start_line=class_def_node.start_point[0] + 1,
                    end_line=class_def_node.end_point[0] + 1,
                    start_col=class_def_node.start_point[1],
                    end_col=class_def_node.end_point[1],
                    content=self._node_text(class_def_node, content),
                ))

                # Extract methods within this class
                method_entities = self._extract_methods(class_def_node, content, class_name)
                entities.extend(method_entities)

        return entities

    def _extract_methods(
        self,
        class_node: Node,
        content: bytes,
        class_name: str
    ) -> List[ParsedEntity]:
        """Extract methods from a class node."""
        entities = []

        # Find function_definition nodes within the class body
        for child in class_node.children:
            if child.type == "block":
                for stmt in child.children:
                    if stmt.type == "function_definition":
                        # Get method name
                        name_node = stmt.child_by_field_name("name")
                        if name_node:
                            method_name = self._node_text(name_node, content)
                            entities.append(ParsedEntity(
                                name=method_name,
                                entity_type="method",
                                start_line=stmt.start_point[0] + 1,
                                end_line=stmt.end_point[0] + 1,
                                start_col=stmt.start_point[1],
                                end_col=stmt.end_point[1],
                                parent=class_name,
                                content=self._node_text(stmt, content),
                            ))

        return entities

    def _extract_functions(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract top-level function definitions (not methods)."""
        entities = []
        query = self._queries["functions"]

        # tree-sitter 0.25+ API: use QueryCursor with captures()
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        func_names = captures.get("func_name", [])
        func_defs = captures.get("func_def", [])

        for func_def_node in func_defs:
            # Check if this is a top-level function (not inside a class)
            if self._is_method(func_def_node):
                continue

            # Find the function name within this definition
            func_name = None
            for name_node in func_names:
                if (name_node.start_byte >= func_def_node.start_byte and
                    name_node.end_byte <= func_def_node.end_byte):
                    func_name = self._node_text(name_node, content)
                    break

            if func_name:
                entities.append(ParsedEntity(
                    name=func_name,
                    entity_type="function",
                    start_line=func_def_node.start_point[0] + 1,
                    end_line=func_def_node.end_point[0] + 1,
                    start_col=func_def_node.start_point[1],
                    end_col=func_def_node.end_point[1],
                    content=self._node_text(func_def_node, content),
                ))

        return entities

    def _extract_imports(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract import statements."""
        entities = []
        query = self._queries["imports"]

        # tree-sitter 0.25+ API: use QueryCursor with captures()
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        seen_imports: Set[str] = set()  # Deduplicate

        # Collect all import/module names from captures
        import_nodes = captures.get("import_name", [])
        module_nodes = captures.get("module_name", [])

        for node in import_nodes + module_nodes:
            import_name = self._node_text(node, content)

            if import_name and import_name not in seen_imports:
                seen_imports.add(import_name)
                entities.append(ParsedEntity(
                    name=import_name,
                    entity_type="import",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    start_col=node.start_point[1],
                    end_col=node.end_point[1],
                ))

        return entities

    def _is_method(self, func_node: Node) -> bool:
        """Check if a function node is a method (inside a class)."""
        parent = func_node.parent
        while parent:
            # Python class
            if parent.type == "class_definition":
                return True
            # TypeScript class
            if parent.type == "class_declaration":
                return True
            parent = parent.parent
        return False

    def _node_text(self, node: Node, content: bytes) -> str:
        """Extract text from a node."""
        return content[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # =========================================================================
    # TypeScript/TSX Specific Extraction Methods
    # =========================================================================

    def _extract_interfaces(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract TypeScript interface declarations."""
        entities = []
        query = self._queries.get("interfaces")
        if not query:
            return entities

        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        interface_names = captures.get("interface_name", [])
        interface_defs = captures.get("interface_def", [])

        for interface_def_node in interface_defs:
            # Find the interface name within this definition
            interface_name = None
            for name_node in interface_names:
                if (name_node.start_byte >= interface_def_node.start_byte and
                    name_node.end_byte <= interface_def_node.end_byte):
                    interface_name = self._node_text(name_node, content)
                    break

            if interface_name:
                entities.append(ParsedEntity(
                    name=interface_name,
                    entity_type="interface",
                    start_line=interface_def_node.start_point[0] + 1,
                    end_line=interface_def_node.end_point[0] + 1,
                    start_col=interface_def_node.start_point[1],
                    end_col=interface_def_node.end_point[1],
                    content=self._node_text(interface_def_node, content),
                ))

        return entities

    def _extract_types(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract TypeScript type alias declarations."""
        entities = []
        query = self._queries.get("types")
        if not query:
            return entities

        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        type_names = captures.get("type_name", [])
        type_defs = captures.get("type_def", [])

        for type_def_node in type_defs:
            # Find the type name within this definition
            type_name = None
            for name_node in type_names:
                if (name_node.start_byte >= type_def_node.start_byte and
                    name_node.end_byte <= type_def_node.end_byte):
                    type_name = self._node_text(name_node, content)
                    break

            if type_name:
                entities.append(ParsedEntity(
                    name=type_name,
                    entity_type="type",
                    start_line=type_def_node.start_point[0] + 1,
                    end_line=type_def_node.end_point[0] + 1,
                    start_col=type_def_node.start_point[1],
                    end_col=type_def_node.end_point[1],
                    content=self._node_text(type_def_node, content),
                ))

        return entities

    def _extract_react_components(self, tree: Tree, content: bytes) -> List[ParsedEntity]:
        """Extract React component definitions (functions returning JSX)."""
        entities = []
        query = self._queries.get("react_components")
        if not query:
            return entities

        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        component_names = captures.get("component_name", [])
        component_defs = captures.get("component_def", [])

        seen_components: Set[str] = set()

        for comp_def_node in component_defs:
            # Find the component name within this definition
            comp_name = None
            for name_node in component_names:
                if (name_node.start_byte >= comp_def_node.start_byte and
                    name_node.end_byte <= comp_def_node.end_byte):
                    comp_name = self._node_text(name_node, content)
                    break

            # React components are PascalCase by convention
            if comp_name and comp_name[0].isupper() and comp_name not in seen_components:
                seen_components.add(comp_name)

                # Check if this contains JSX
                comp_content = self._node_text(comp_def_node, content)
                if "<" in comp_content and (">" in comp_content or "/>" in comp_content):
                    entities.append(ParsedEntity(
                        name=comp_name,
                        entity_type="react_component",
                        start_line=comp_def_node.start_point[0] + 1,
                        end_line=comp_def_node.end_point[0] + 1,
                        start_col=comp_def_node.start_point[1],
                        end_col=comp_def_node.end_point[1],
                        content=comp_content,
                        metadata={"is_component": True},
                    ))

        return entities

    def _entities_to_graph(
        self,
        entities: List[ParsedEntity],
        module_name: str,
        file_path: Optional[str],
        content: bytes,
    ) -> Tuple[List[NodeData], List[EdgeData]]:
        """Convert parsed entities to graph nodes and edges."""
        nodes = []
        edges = []

        # Create module node
        module_id = f"module::{module_name}"
        module_node = NodeData(
            id=module_id,
            type=NodeType.CODE.value,
            content="",  # Don't store full file content
            status=NodeStatus.VERIFIED.value,
            data={
                "kind": "module",
                "name": module_name,
                "path": file_path,
                "lines": content.count(b"\n") + 1,
            },
        )
        nodes.append(module_node)

        # Track entity IDs for edge creation
        entity_ids: Dict[str, str] = {}

        for entity in entities:
            # Generate unique ID
            if entity.entity_type == "method":
                entity_id = f"{entity.entity_type}::{module_name}::{entity.parent}::{entity.name}"
            else:
                entity_id = f"{entity.entity_type}::{module_name}::{entity.name}"

            entity_ids[entity.name] = entity_id

            # Map entity type to NodeType
            node_type = self._entity_type_to_node_type(entity.entity_type)

            # Create node
            node = NodeData(
                id=entity_id,
                type=node_type,
                content=entity.content or "",
                status=NodeStatus.VERIFIED.value,
                data={
                    "kind": entity.entity_type,
                    "name": entity.name,
                    "start_line": entity.start_line,
                    "end_line": entity.end_line,
                    "parent": entity.parent,
                },
            )
            nodes.append(node)

            # Create CONTAINS edge (module contains entity, or class contains method)
            if entity.entity_type == "method" and entity.parent:
                # Method is contained by class
                parent_id = f"class::{module_name}::{entity.parent}"
                if any(n.id == parent_id for n in nodes):
                    edges.append(EdgeData(
                        source_id=parent_id,
                        target_id=entity_id,
                        type=EdgeType.CONTAINS.value,
                    ))
            elif entity.entity_type != "import":
                # Top-level entity is contained by module
                edges.append(EdgeData(
                    source_id=module_id,
                    target_id=entity_id,
                    type=EdgeType.CONTAINS.value,
                ))

            # Create import edges
            if entity.entity_type == "import":
                edges.append(EdgeData(
                    source_id=module_id,
                    target_id=entity_id,
                    type=EdgeType.REFERENCES.value,
                    metadata={"import_name": entity.name},
                ))

        return nodes, edges

    def _entity_type_to_node_type(self, entity_type: str) -> str:
        """Map parsed entity type to NodeType enum value."""
        mapping = {
            # Python types
            "class": NodeType.CLASS.value,
            "function": NodeType.FUNCTION.value,
            "method": NodeType.FUNCTION.value,
            "import": NodeType.CODE.value,  # Imports are code references
            "call": NodeType.CALL.value,
            # TypeScript types
            "interface": NodeType.SPEC.value,  # Interfaces are specifications
            "type": NodeType.SPEC.value,       # Type aliases are specifications
            "react_component": NodeType.CODE.value,  # Components are code
        }
        return mapping.get(entity_type, NodeType.CODE.value)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_python_file(path: Path) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Parse a Python file and return graph nodes/edges.

    Convenience wrapper around CodeParser.
    """
    parser = CodeParser(language="python")
    return parser.parse_file(path)


def parse_python_directory(
    directory: Path,
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Parse all Python files in a directory.

    Args:
        directory: Root directory to scan
        recursive: Whether to recurse into subdirectories
        exclude_patterns: Patterns to exclude (e.g., ["__pycache__", ".venv"])

    Returns:
        Combined (nodes, edges) from all files
    """
    directory = Path(directory)
    exclude_patterns = exclude_patterns or ["__pycache__", ".venv", "venv", ".git", ".egg"]

    all_nodes = []
    all_edges = []

    parser = CodeParser(language="python")

    # Find Python files
    pattern = "**/*.py" if recursive else "*.py"

    for py_file in directory.glob(pattern):
        # Check exclusions
        if any(excl in str(py_file) for excl in exclude_patterns):
            continue

        try:
            nodes, edges = parser.parse_file(py_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as e:
            # Log but continue on parse errors
            print(f"Warning: Failed to parse {py_file}: {e}")

    return all_nodes, all_edges


def parse_typescript_file(path: Path) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Parse a TypeScript file and return graph nodes/edges.

    Convenience wrapper around CodeParser.
    """
    parser = CodeParser(language="typescript")
    return parser.parse_file(path)


def parse_tsx_file(path: Path) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Parse a TSX (React) file and return graph nodes/edges.

    Convenience wrapper around CodeParser.
    """
    parser = CodeParser(language="tsx")
    return parser.parse_file(path)


def parse_typescript_directory(
    directory: Path,
    recursive: bool = True,
    exclude_patterns: Optional[List[str]] = None,
) -> Tuple[List[NodeData], List[EdgeData]]:
    """
    Parse all TypeScript/TSX files in a directory.

    Args:
        directory: Root directory to scan
        recursive: Whether to recurse into subdirectories
        exclude_patterns: Patterns to exclude (e.g., ["node_modules", "dist"])

    Returns:
        Combined (nodes, edges) from all files
    """
    directory = Path(directory)
    exclude_patterns = exclude_patterns or ["node_modules", "dist", "build", ".git", "coverage"]

    all_nodes = []
    all_edges = []

    # Parse .ts files
    ts_parser = CodeParser(language="typescript")
    tsx_parser = CodeParser(language="tsx")

    pattern_prefix = "**/" if recursive else ""

    # TypeScript files
    for ts_file in directory.glob(f"{pattern_prefix}*.ts"):
        if any(excl in str(ts_file) for excl in exclude_patterns):
            continue
        # Skip .d.ts declaration files
        if ts_file.name.endswith(".d.ts"):
            continue
        try:
            nodes, edges = ts_parser.parse_file(ts_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as e:
            print(f"Warning: Failed to parse {ts_file}: {e}")

    # TSX files
    for tsx_file in directory.glob(f"{pattern_prefix}*.tsx"):
        if any(excl in str(tsx_file) for excl in exclude_patterns):
            continue
        try:
            nodes, edges = tsx_parser.parse_file(tsx_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as e:
            print(f"Warning: Failed to parse {tsx_file}: {e}")

    # JSX files (use TSX parser)
    for jsx_file in directory.glob(f"{pattern_prefix}*.jsx"):
        if any(excl in str(jsx_file) for excl in exclude_patterns):
            continue
        try:
            nodes, edges = tsx_parser.parse_file(jsx_file)
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as e:
            print(f"Warning: Failed to parse {jsx_file}: {e}")

    return all_nodes, all_edges


def ingest_codebase(
    directory: Path,
    db,  # ParagonDB - avoid circular import
    recursive: bool = True,
    languages: Optional[List[str]] = None,
) -> Tuple[int, int]:
    """
    Parse and ingest an entire codebase into ParagonDB.

    Args:
        directory: Root directory
        db: ParagonDB instance
        recursive: Whether to recurse
        languages: List of languages to parse (default: ["python"])
                   Options: "python", "typescript"

    Returns:
        Tuple of (nodes_added, edges_added)
    """
    languages = languages or ["python"]

    all_nodes = []
    all_edges = []

    if "python" in languages:
        nodes, edges = parse_python_directory(directory, recursive=recursive)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    if "typescript" in languages:
        nodes, edges = parse_typescript_directory(directory, recursive=recursive)
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    db.add_nodes_batch(all_nodes)

    # Only add edges where both nodes exist
    valid_node_ids = {n.id for n in all_nodes}
    valid_edges = [
        e for e in all_edges
        if e.source_id in valid_node_ids and e.target_id in valid_node_ids
    ]

    db.add_edges_batch(valid_edges)

    return len(all_nodes), len(valid_edges)
