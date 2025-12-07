"""
PARAGON AGENT TOOLS - LangGraph Tool Definitions

Tools for LangGraph agents to interact with ParagonDB.

Design:
- Each tool is a typed function with clear input/output schemas
- Tools are stateless - they operate on the global ParagonDB instance
- All tools return structured results (msgspec-compatible)

Tool Categories:
1. Graph Operations: add_node, add_edge, query_nodes, query_edges
2. Analysis: get_waves, get_descendants, get_ancestors, check_cycle
3. Parsing: parse_source_code
4. Alignment: align_graphs
5. Layer 7B (Auditor): add_node_safe, verify_alignment, check_syntax

Layer 7 Architecture:
- Layer 7A (Creator): LLM generates structures via core/llm.py
- Layer 7B (Auditor): This module provides guardrails before graph insertion
"""
from typing import List, Dict, Any, Optional, Annotated, Tuple, Callable
from pathlib import Path
import msgspec

from core.graph_db import ParagonDB, DuplicateNodeError, TopologyViolationError
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from domain.code_parser import CodeParser, parse_python_directory

# Data Loader for bulk import (optional)
try:
    from infrastructure.data_loader import (
        PolarsLoader,
        BulkIngestor,
        load_graph_from_parquet,
        load_graph_from_csv,
    )
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False


# =============================================================================
# GLOBAL STATE
# =============================================================================

_db: Optional[ParagonDB] = None
_parser: Optional[CodeParser] = None
_mutation_logger = None  # Lazy-initialized MutationLogger
_git_sync = None  # Lazy-initialized GitSync
_pending_transaction: List[tuple] = []  # Collect mutations for git commit


def _get_mutation_logger():
    """Get or create the mutation logger (lazy initialization)."""
    global _mutation_logger
    if _mutation_logger is None:
        try:
            from infrastructure.logger import MutationLogger
            _mutation_logger = MutationLogger()
        except ImportError:
            pass  # Logger not available
    return _mutation_logger


def _get_git_sync():
    """Get or create the GitSync instance (lazy initialization)."""
    global _git_sync
    if _git_sync is None:
        try:
            from infrastructure.git_sync import GitSync, load_git_config
            config = load_git_config(_db)
            _git_sync = GitSync(config=config, db=_db)
        except ImportError:
            pass  # GitSync not available
    return _git_sync


def _record_transaction(node_id: str, node_type: str, edge_info: Optional[tuple] = None):
    """Record a mutation for the current transaction batch."""
    global _pending_transaction
    _pending_transaction.append(("node", node_id, node_type))
    if edge_info:
        _pending_transaction.append(("edge", edge_info[0], edge_info[1], edge_info[2]))


def flush_transaction(agent_id: str = "agent"):
    """
    Flush pending mutations to GitSync for commit.

    Call this at transaction boundaries (e.g., after a batch of related changes).
    """
    global _pending_transaction
    git_sync = _get_git_sync()
    if git_sync and _pending_transaction:
        try:
            nodes_created = [t[1] for t in _pending_transaction if t[0] == "node"]
            edges_created = [(t[1], t[2], t[3]) for t in _pending_transaction if t[0] == "edge"]
            git_sync.on_transaction_complete(
                nodes_created=nodes_created,
                edges_created=edges_created,
                agent_id=agent_id,
            )
        except Exception:
            pass  # Don't fail on git errors
        finally:
            _pending_transaction = []


def _log_node_created(node_id: str, node_type: str, created_by: str):
    """Log a node creation event and record for transaction."""
    # Log to MutationLogger
    logger = _get_mutation_logger()
    if logger:
        try:
            logger.log_node_created(
                node_id=node_id,
                node_type=node_type,
                agent_id=created_by,
            )
        except Exception:
            pass  # Don't fail on logging errors

    # Record for GitSync transaction
    _record_transaction(node_id, node_type)


def _log_edge_created(source_id: str, target_id: str, edge_type: str):
    """Log an edge creation event and record for transaction."""
    # Log to MutationLogger
    logger = _get_mutation_logger()
    if logger:
        try:
            logger.log_edge_created(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
            )
        except Exception:
            pass

    # Record for GitSync transaction
    global _pending_transaction
    _pending_transaction.append(("edge", source_id, target_id, edge_type))


def get_db() -> ParagonDB:
    """Get or create the global ParagonDB instance."""
    global _db
    if _db is None:
        _db = ParagonDB()
    return _db


def set_db(db: ParagonDB) -> None:
    """
    Set the global ParagonDB instance.

    IMPORTANT: This is the dependency injection point for tests.
    Call this BEFORE running any agent/tool operations to ensure
    they operate on your test database, not a default instance.

    Example:
        def test_my_agent():
            test_db = ParagonDB()
            set_db(test_db)  # Inject test DB
            try:
                result = add_node("CODE", "test content")
                assert result.success
            finally:
                set_db(None)  # Clean up (optional)
    """
    global _db
    _db = db


def get_parser() -> CodeParser:
    """Get or create the global CodeParser instance."""
    global _parser
    if _parser is None:
        _parser = CodeParser()
    return _parser


# =============================================================================
# RESULT TYPES (msgspec Structs for typed returns)
# =============================================================================

class ToolResult(msgspec.Struct):
    """Base result type for all tools."""
    success: bool
    message: str


class NodeResult(msgspec.Struct):
    """Result of node creation."""
    success: bool
    node_id: str
    message: str


class BatchNodeResult(msgspec.Struct):
    """Result of batch node creation."""
    success: bool
    node_ids: List[str]
    count: int
    message: str


class EdgeResult(msgspec.Struct):
    """Result of edge creation."""
    success: bool
    count: int
    message: str


class QueryResult(msgspec.Struct):
    """Result of a query operation."""
    success: bool
    count: int
    node_ids: List[str]
    message: str


class WaveResult(msgspec.Struct):
    """Result of wavefront analysis."""
    success: bool
    layer_count: int
    layers: List[List[str]]
    message: str


class CycleResult(msgspec.Struct):
    """Result of cycle detection."""
    success: bool
    has_cycle: bool
    message: str


class ParseResult(msgspec.Struct):
    """Result of source parsing."""
    success: bool
    nodes_added: int
    edges_added: int
    path: str
    message: str


class AlignmentResult(msgspec.Struct):
    """Result of graph alignment."""
    success: bool
    score: float
    mappings: Dict[str, str]
    unmapped_source: List[str]
    unmapped_target: List[str]
    message: str


class SyntaxCheckResult(msgspec.Struct):
    """Result of syntax validation via tree-sitter."""
    success: bool
    valid: bool
    language: str
    errors: List[str]
    warnings: List[str]
    message: str


class AuditResult(msgspec.Struct):
    """Combined result from Layer 7B auditor checks."""
    success: bool
    node_id: str
    syntax_valid: bool
    topology_valid: bool
    alignment_score: float
    violations: List[str]
    approved: bool
    message: str


class SafeNodeResult(msgspec.Struct):
    """Result of add_node_safe (with auditor checks)."""
    success: bool
    node_id: str
    syntax_valid: bool
    topology_valid: bool
    violations: List[str]
    message: str


# =============================================================================
# GRAPH OPERATIONS
# =============================================================================

def add_node(
    node_type: Annotated[str, "Node type from ontology (CODE, SPEC, TEST, etc.)"],
    content: Annotated[str, "Node content (source code, specification text, etc.)"],
    data: Annotated[Optional[Dict[str, Any]], "Additional metadata"] = None,
    created_by: Annotated[str, "Creator identifier"] = "agent",
) -> NodeResult:
    """
    Add a single node to the graph.

    Args:
        node_type: Type of node (CODE, SPEC, TEST, DOC, CONFIG, SCHEMA, etc.)
        content: The content of the node
        data: Optional metadata dictionary
        created_by: Who/what created this node

    Returns:
        NodeResult with success status and node ID
    """
    db = get_db()

    try:
        node = NodeData.create(
            type=node_type,
            content=content,
            data=data or {},
            created_by=created_by,
        )
        db.add_node(node)
        return NodeResult(
            success=True,
            node_id=node.id,
            message=f"Created node {node.id} of type {node_type}"
        )
    except DuplicateNodeError as e:
        return NodeResult(
            success=False,
            node_id="",
            message=str(e)
        )
    except Exception as e:
        return NodeResult(
            success=False,
            node_id="",
            message=f"Failed to create node: {e}"
        )


def add_nodes_batch(
    nodes: Annotated[List[Dict[str, Any]], "List of node specifications"],
) -> BatchNodeResult:
    """
    Add multiple nodes to the graph in a batch.

    Args:
        nodes: List of dicts with keys: type, content, data (optional), created_by (optional)

    Returns:
        BatchNodeResult with success status and node IDs
    """
    db = get_db()

    try:
        node_objects = []
        for item in nodes:
            node = NodeData.create(
                type=item.get("type", NodeType.CODE.value),
                content=item.get("content", ""),
                data=item.get("data", {}),
                created_by=item.get("created_by", "agent"),
            )
            node_objects.append(node)

        db.add_nodes_batch(node_objects)
        node_ids = [n.id for n in node_objects]

        return BatchNodeResult(
            success=True,
            node_ids=node_ids,
            count=len(node_objects),
            message=f"Created {len(node_objects)} nodes"
        )
    except Exception as e:
        return BatchNodeResult(
            success=False,
            node_ids=[],
            count=0,
            message=f"Failed to create nodes: {e}"
        )


def add_edge(
    source_id: Annotated[str, "Source node ID"],
    target_id: Annotated[str, "Target node ID"],
    edge_type: Annotated[str, "Edge type (DEPENDS_ON, IMPLEMENTS, TESTS, etc.)"],
    weight: Annotated[float, "Edge weight"] = 1.0,
    metadata: Annotated[Optional[Dict[str, Any]], "Additional metadata"] = None,
) -> EdgeResult:
    """
    Add an edge between two nodes.

    Args:
        source_id: ID of the source node
        target_id: ID of the target node
        edge_type: Type of relationship
        weight: Edge weight (default 1.0)
        metadata: Optional metadata

    Returns:
        EdgeResult with success status
    """
    db = get_db()

    try:
        edge = EdgeData.create(
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            weight=weight,
            metadata=metadata or {},
        )
        db.add_edge(edge)
        return EdgeResult(
            success=True,
            count=1,
            message=f"Created edge {source_id} -> {target_id} ({edge_type})"
        )
    except Exception as e:
        return EdgeResult(
            success=False,
            count=0,
            message=f"Failed to create edge: {e}"
        )


def add_edges_batch(
    edges: Annotated[List[Dict[str, Any]], "List of edge specifications"],
) -> EdgeResult:
    """
    Add multiple edges to the graph in a batch.

    Args:
        edges: List of dicts with keys: source_id, target_id, type, weight (optional), metadata (optional)

    Returns:
        EdgeResult with success status and count
    """
    db = get_db()

    try:
        edge_objects = []
        for item in edges:
            edge = EdgeData.create(
                source_id=item["source_id"],
                target_id=item["target_id"],
                type=item.get("type", EdgeType.DEPENDS_ON.value),
                weight=item.get("weight", 1.0),
                metadata=item.get("metadata", {}),
            )
            edge_objects.append(edge)

        db.add_edges_batch(edge_objects)

        return EdgeResult(
            success=True,
            count=len(edge_objects),
            message=f"Created {len(edge_objects)} edges"
        )
    except Exception as e:
        return EdgeResult(
            success=False,
            count=0,
            message=f"Failed to create edges: {e}"
        )


def query_nodes(
    node_type: Annotated[Optional[str], "Filter by node type"] = None,
    status: Annotated[Optional[str], "Filter by status"] = None,
    limit: Annotated[int, "Maximum results"] = 100,
) -> QueryResult:
    """
    Query nodes with optional filters.

    Args:
        node_type: Optional type filter
        status: Optional status filter
        limit: Maximum number of results

    Returns:
        QueryResult with matching node IDs
    """
    db = get_db()

    try:
        nodes = db.get_all_nodes()

        if node_type:
            nodes = [n for n in nodes if n.type == node_type]
        if status:
            nodes = [n for n in nodes if n.status == status]

        nodes = nodes[:limit]
        node_ids = [n.id for n in nodes]

        return QueryResult(
            success=True,
            count=len(node_ids),
            node_ids=node_ids,
            message=f"Found {len(node_ids)} nodes"
        )
    except Exception as e:
        return QueryResult(
            success=False,
            count=0,
            node_ids=[],
            message=f"Query failed: {e}"
        )


def get_node(
    node_id: Annotated[str, "Node ID to retrieve"],
) -> Dict[str, Any]:
    """
    Get a single node by ID.

    Args:
        node_id: The node ID

    Returns:
        Node data as dictionary, or error dict
    """
    db = get_db()

    node = db.get_node(node_id)
    if node is None:
        return {"error": f"Node not found: {node_id}"}

    return {
        "id": node.id,
        "type": node.type,
        "status": node.status,
        "content": node.content,
        "data": node.data,
        "version": node.version,
        "created_at": node.created_at,
        "created_by": node.created_by,
    }


# =============================================================================
# ANALYSIS OPERATIONS
# =============================================================================

def get_waves() -> WaveResult:
    """
    Get wavefront layers (topological sort into parallel layers).

    Nodes in the same layer have no dependencies on each other
    and can be processed in parallel.

    Returns:
        WaveResult with layers of node IDs
    """
    db = get_db()

    try:
        waves = db.get_waves()
        layers = [[node.id for node in layer] for layer in waves]

        return WaveResult(
            success=True,
            layer_count=len(layers),
            layers=layers,
            message=f"Graph has {len(layers)} wavefront layers"
        )
    except Exception as e:
        return WaveResult(
            success=False,
            layer_count=0,
            layers=[],
            message=f"Wavefront analysis failed: {e}"
        )


def get_descendants(
    node_id: Annotated[str, "Node ID to get descendants of"],
) -> QueryResult:
    """
    Get all descendants of a node (nodes reachable from this node).

    Args:
        node_id: The starting node ID

    Returns:
        QueryResult with descendant node IDs
    """
    db = get_db()

    try:
        descendants = db.get_descendants(node_id)
        desc_ids = [d.id for d in descendants]

        return QueryResult(
            success=True,
            count=len(desc_ids),
            node_ids=desc_ids,
            message=f"Node {node_id} has {len(desc_ids)} descendants"
        )
    except KeyError:
        return QueryResult(
            success=False,
            count=0,
            node_ids=[],
            message=f"Node not found: {node_id}"
        )
    except Exception as e:
        return QueryResult(
            success=False,
            count=0,
            node_ids=[],
            message=f"Failed to get descendants: {e}"
        )


def get_ancestors(
    node_id: Annotated[str, "Node ID to get ancestors of"],
) -> QueryResult:
    """
    Get all ancestors of a node (nodes that can reach this node).

    Args:
        node_id: The starting node ID

    Returns:
        QueryResult with ancestor node IDs
    """
    db = get_db()

    try:
        ancestors = db.get_ancestors(node_id)
        anc_ids = [a.id for a in ancestors]

        return QueryResult(
            success=True,
            count=len(anc_ids),
            node_ids=anc_ids,
            message=f"Node {node_id} has {len(anc_ids)} ancestors"
        )
    except KeyError:
        return QueryResult(
            success=False,
            count=0,
            node_ids=[],
            message=f"Node not found: {node_id}"
        )
    except Exception as e:
        return QueryResult(
            success=False,
            count=0,
            node_ids=[],
            message=f"Failed to get ancestors: {e}"
        )


def check_cycle() -> CycleResult:
    """
    Check if the graph contains any cycles.

    Returns:
        CycleResult with cycle status
    """
    db = get_db()

    try:
        has_cycle = db.has_cycle()
        return CycleResult(
            success=True,
            has_cycle=has_cycle,
            message="Graph contains a cycle" if has_cycle else "Graph is acyclic (DAG)"
        )
    except Exception as e:
        return CycleResult(
            success=False,
            has_cycle=False,
            message=f"Cycle detection failed: {e}"
        )


def get_graph_stats() -> Dict[str, Any]:
    """
    Get graph statistics.

    Returns:
        Dictionary with node_count, edge_count, has_cycle, is_empty
    """
    db = get_db()

    return {
        "node_count": db.node_count,
        "edge_count": db.edge_count,
        "has_cycle": db.has_cycle(),
        "is_empty": db.is_empty,
    }


# =============================================================================
# PARSING OPERATIONS
# =============================================================================

def parse_source(
    path: Annotated[str, "Path to file or directory to parse"],
    recursive: Annotated[bool, "Recursively parse directories"] = True,
) -> ParseResult:
    """
    Parse source code and add to graph.

    Args:
        path: Path to Python file or directory
        recursive: Whether to recursively parse directories

    Returns:
        ParseResult with counts of nodes and edges added
    """
    db = get_db()
    parser = get_parser()

    path_obj = Path(path)
    if not path_obj.exists():
        return ParseResult(
            success=False,
            nodes_added=0,
            edges_added=0,
            path=str(path),
            message=f"Path not found: {path}"
        )

    try:
        if path_obj.is_file():
            nodes, edges = parser.parse_file(path_obj)
        else:
            nodes, edges = parse_python_directory(path_obj, recursive=recursive)

        # Add nodes to database
        db.add_nodes_batch(nodes)

        # Only add edges where both nodes exist
        valid_node_ids = {n.id for n in nodes}
        valid_edges = [
            e for e in edges
            if e.source_id in valid_node_ids and e.target_id in valid_node_ids
        ]
        db.add_edges_batch(valid_edges)

        return ParseResult(
            success=True,
            nodes_added=len(nodes),
            edges_added=len(valid_edges),
            path=str(path),
            message=f"Parsed {path}: {len(nodes)} nodes, {len(valid_edges)} edges"
        )
    except Exception as e:
        return ParseResult(
            success=False,
            nodes_added=0,
            edges_added=0,
            path=str(path),
            message=f"Parse failed: {e}"
        )


# =============================================================================
# BULK IMPORT/EXPORT OPERATIONS
# =============================================================================

class BulkImportResult(msgspec.Struct, kw_only=True):
    """Result of a bulk import operation."""
    success: bool
    nodes_imported: int = 0
    edges_imported: int = 0
    format: str = ""
    message: str = ""


def import_graph_from_file(
    nodes_path: Annotated[str, "Path to nodes file (CSV, Parquet, or Arrow)"],
    edges_path: Annotated[Optional[str], "Path to edges file (optional)"] = None,
    format: Annotated[str, "File format: csv, parquet, or arrow"] = "parquet",
) -> BulkImportResult:
    """
    Bulk import graph data from files.

    Efficiently populates the graph from CSV, Parquet, or Arrow files.
    Uses Polars lazy evaluation for optimal performance with large datasets.

    Args:
        nodes_path: Path to nodes file
        edges_path: Optional path to edges file
        format: File format (csv, parquet, arrow)

    Returns:
        BulkImportResult with import counts
    """
    if not DATA_LOADER_AVAILABLE:
        return BulkImportResult(
            success=False,
            message="Data loader not available (infrastructure.data_loader not found)"
        )

    db = get_db()
    nodes_path_obj = Path(nodes_path)

    if not nodes_path_obj.exists():
        return BulkImportResult(
            success=False,
            message=f"Nodes file not found: {nodes_path}"
        )

    try:
        ingestor = BulkIngestor(db)
        nodes_count, edges_count = ingestor.ingest_from_files(
            nodes_path=nodes_path,
            edges_path=edges_path,
            format=format,
        )

        return BulkImportResult(
            success=True,
            nodes_imported=nodes_count,
            edges_imported=edges_count,
            format=format,
            message=f"Imported {nodes_count} nodes and {edges_count} edges from {format} files"
        )
    except Exception as e:
        return BulkImportResult(
            success=False,
            format=format,
            message=f"Import failed: {e}"
        )


def export_graph_to_parquet(
    nodes_path: Annotated[str, "Output path for nodes.parquet"],
    edges_path: Annotated[str, "Output path for edges.parquet"],
) -> Dict[str, Any]:
    """
    Export graph data to Parquet files.

    Parquet is efficient for large datasets:
    - Columnar storage
    - Built-in compression
    - Fast read/write via Polars

    Args:
        nodes_path: Output path for nodes
        edges_path: Output path for edges

    Returns:
        Dict with export results
    """
    import polars as pl

    db = get_db()

    try:
        # Export nodes
        nodes = list(db.query_nodes())
        if nodes:
            nodes_data = [{
                "id": n.id,
                "type": n.type,
                "content": n.content,
                "status": n.status,
                "created_by": n.created_by,
            } for n in nodes]
            nodes_df = pl.DataFrame(nodes_data)
            nodes_df.write_parquet(nodes_path)

        # Export edges
        edges = list(db.query_edges())
        if edges:
            edges_data = [{
                "source_id": e.source_id,
                "target_id": e.target_id,
                "type": e.type,
                "weight": e.weight,
            } for e in edges]
            edges_df = pl.DataFrame(edges_data)
            edges_df.write_parquet(edges_path)

        return {
            "success": True,
            "nodes_exported": len(nodes),
            "edges_exported": len(edges),
            "nodes_path": nodes_path,
            "edges_path": edges_path,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# ALIGNMENT OPERATIONS
# =============================================================================

def align_node_sets(
    source_ids: Annotated[List[str], "Source node IDs"],
    target_ids: Annotated[List[str], "Target node IDs"],
    algorithm: Annotated[str, "Matching algorithm (rrwm, ipfp, sm, hungarian)"] = "rrwm",
) -> AlignmentResult:
    """
    Align two sets of nodes using graph matching.

    Args:
        source_ids: IDs of source nodes
        target_ids: IDs of target nodes
        algorithm: Matching algorithm to use

    Returns:
        AlignmentResult with mappings and scores
    """
    from core.alignment import GraphAligner, MatchingAlgorithm

    db = get_db()

    try:
        # Get nodes
        source_nodes = [db.get_node(nid) for nid in source_ids]
        source_nodes = [n for n in source_nodes if n is not None]

        target_nodes = [db.get_node(nid) for nid in target_ids]
        target_nodes = [n for n in target_nodes if n is not None]

        if not source_nodes or not target_nodes:
            return AlignmentResult(
                success=False,
                score=0.0,
                mappings={},
                unmapped_source=[],
                unmapped_target=[],
                message="No valid nodes found"
            )

        # Get edges within each set
        source_id_set = set(source_ids)
        target_id_set = set(target_ids)

        all_edges = db.get_all_edges()
        source_edges = [
            e for e in all_edges
            if e.source_id in source_id_set and e.target_id in source_id_set
        ]
        target_edges = [
            e for e in all_edges
            if e.source_id in target_id_set and e.target_id in target_id_set
        ]

        # Align
        algo = MatchingAlgorithm(algorithm)
        aligner = GraphAligner(algorithm=algo)
        result = aligner.align(source_nodes, source_edges, target_nodes, target_edges)

        return AlignmentResult(
            success=True,
            score=result.score,
            mappings=result.node_mapping,
            unmapped_source=result.unmapped_source,
            unmapped_target=result.unmapped_target,
            message=f"Alignment score: {result.score:.3f}"
        )
    except ValueError as e:
        return AlignmentResult(
            success=False,
            score=0.0,
            mappings={},
            unmapped_source=[],
            unmapped_target=[],
            message=f"Invalid algorithm: {e}"
        )
    except Exception as e:
        return AlignmentResult(
            success=False,
            score=0.0,
            mappings={},
            unmapped_source=[],
            unmapped_target=[],
            message=f"Alignment failed: {e}"
        )


# =============================================================================
# LAYER 7B - THE AUDITOR (Safety Hooks)
# =============================================================================

def check_syntax(
    code: Annotated[str, "Code to check"],
    language: Annotated[str, "Language (python, javascript, etc.)"] = "python",
) -> SyntaxCheckResult:
    """
    Check code syntax using tree-sitter.

    This is the first layer of the auditor - no code enters the graph
    without passing syntax validation.

    Args:
        code: Source code to validate
        language: Programming language

    Returns:
        SyntaxCheckResult with validity and any errors
    """
    try:
        # Import tree-sitter dynamically to avoid hard dependency at module level
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser

        errors = []
        warnings = []

        if language == "python":
            PYTHON_LANGUAGE = Language(tspython.language())
            parser = Parser(PYTHON_LANGUAGE)

            tree = parser.parse(bytes(code, "utf8"))

            # Check for syntax errors in the tree
            if tree.root_node.has_error:
                # Walk tree to find error nodes
                def find_errors(node, path=""):
                    if node.is_error or node.is_missing:
                        line = node.start_point[0] + 1
                        col = node.start_point[1] + 1
                        errors.append(f"Syntax error at line {line}, column {col}")
                    for child in node.children:
                        find_errors(child, path + "/" + node.type)

                find_errors(tree.root_node)

            valid = len(errors) == 0

            return SyntaxCheckResult(
                success=True,
                valid=valid,
                language=language,
                errors=errors,
                warnings=warnings,
                message="Syntax check passed" if valid else f"Found {len(errors)} syntax error(s)"
            )
        else:
            # Unsupported language - pass through with warning
            return SyntaxCheckResult(
                success=True,
                valid=True,
                language=language,
                errors=[],
                warnings=[f"Language '{language}' syntax checking not implemented - skipped"],
                message=f"Syntax check skipped for {language}"
            )

    except ImportError as e:
        return SyntaxCheckResult(
            success=False,
            valid=False,
            language=language,
            errors=[f"tree-sitter not available: {e}"],
            warnings=[],
            message="Syntax check failed - tree-sitter not installed"
        )
    except Exception as e:
        return SyntaxCheckResult(
            success=False,
            valid=False,
            language=language,
            errors=[str(e)],
            warnings=[],
            message=f"Syntax check failed: {e}"
        )


def verify_alignment(
    spec_node_id: Annotated[str, "SPEC node ID"],
    code_node_id: Annotated[str, "CODE node ID"],
    threshold: Annotated[float, "Minimum alignment score (0.0-1.0)"] = 0.6,
) -> AuditResult:
    """
    Verify that generated code aligns with its specification.

    This is the core of Layer 7B - ensuring that what was generated
    actually implements what was specified.

    Uses graph alignment (pygmtools) to compute structural similarity
    between the spec's expected components and the code's actual structure.

    Args:
        spec_node_id: The SPEC node the code should implement
        code_node_id: The CODE node to verify
        threshold: Minimum alignment score to approve

    Returns:
        AuditResult with alignment score and approval status
    """
    db = get_db()

    try:
        # Get both nodes
        spec_node = db.get_node(spec_node_id)
        code_node = db.get_node(code_node_id)

        if spec_node is None:
            return AuditResult(
                success=False,
                node_id=code_node_id,
                syntax_valid=False,
                topology_valid=False,
                alignment_score=0.0,
                violations=[f"SPEC node not found: {spec_node_id}"],
                approved=False,
                message="Verification failed - SPEC not found"
            )

        if code_node is None:
            return AuditResult(
                success=False,
                node_id=code_node_id,
                syntax_valid=False,
                topology_valid=False,
                alignment_score=0.0,
                violations=[f"CODE node not found: {code_node_id}"],
                approved=False,
                message="Verification failed - CODE not found"
            )

        violations = []

        # Step 1: Check syntax of the code
        syntax_result = check_syntax(code_node.content, "python")
        syntax_valid = syntax_result.valid
        if not syntax_valid:
            violations.extend(syntax_result.errors)

        # Step 2: Check topology constraints
        topology_violations = db.validate_topology(code_node_id, mode="hard")
        topology_valid = len(topology_violations) == 0
        violations.extend(topology_violations)

        # Step 3: Compute alignment score
        # For now, use a simple heuristic based on content overlap
        # TODO: Use full pygmtools alignment when components are parsed
        alignment_score = _compute_alignment_score(spec_node, code_node)

        if alignment_score < threshold:
            violations.append(
                f"Alignment score {alignment_score:.2f} below threshold {threshold:.2f}"
            )

        # Determine approval
        approved = syntax_valid and topology_valid and alignment_score >= threshold

        return AuditResult(
            success=True,
            node_id=code_node_id,
            syntax_valid=syntax_valid,
            topology_valid=topology_valid,
            alignment_score=alignment_score,
            violations=violations,
            approved=approved,
            message="Verification passed" if approved else f"Verification failed: {len(violations)} issue(s)"
        )

    except Exception as e:
        return AuditResult(
            success=False,
            node_id=code_node_id,
            syntax_valid=False,
            topology_valid=False,
            alignment_score=0.0,
            violations=[str(e)],
            approved=False,
            message=f"Verification error: {e}"
        )


def _compute_alignment_score(spec_node: NodeData, code_node: NodeData) -> float:
    """
    Compute alignment score between spec and code.

    This is a placeholder for full pygmtools integration.
    Currently uses keyword overlap as a proxy for alignment.

    Args:
        spec_node: The specification node
        code_node: The code node

    Returns:
        Alignment score between 0.0 and 1.0
    """
    import re

    # Extract keywords from spec (function names, class names, etc.)
    spec_text = spec_node.content.lower()
    code_text = code_node.content.lower()

    # Simple word tokenization
    spec_words = set(re.findall(r'\b[a-z_][a-z0-9_]*\b', spec_text))
    code_words = set(re.findall(r'\b[a-z_][a-z0-9_]*\b', code_text))

    # Filter out common words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'and', 'or', 'in', 'on'}
    spec_words = spec_words - stop_words
    code_words = code_words - stop_words

    if not spec_words:
        return 0.5  # No keywords to match - neutral score

    # Jaccard similarity
    intersection = len(spec_words & code_words)
    union = len(spec_words | code_words)

    if union == 0:
        return 0.5

    return intersection / union


def add_node_safe(
    node_type: Annotated[str, "Node type from ontology (CODE, SPEC, TEST, etc.)"],
    content: Annotated[str, "Node content (source code, specification text, etc.)"],
    spec_id: Annotated[Optional[str], "SPEC node ID (for CODE nodes, enables alignment check)"] = None,
    data: Annotated[Optional[Dict[str, Any]], "Additional metadata"] = None,
    created_by: Annotated[str, "Creator identifier"] = "agent",
    check_alignment: Annotated[bool, "Whether to check alignment with spec"] = True,
    signature: Annotated[Optional[Any], "Agent signature for audit trail (AgentSignature)"] = None,
) -> SafeNodeResult:
    """
    Add a node to the graph with Layer 7B auditor checks.

    This is the SAFE version of add_node that performs:
    1. Syntax validation (for CODE nodes)
    2. Topology constraint checking
    3. Alignment verification (if spec_id provided)

    Nodes that fail auditor checks are NOT added to the graph.

    Args:
        node_type: Type of node (CODE, SPEC, TEST, etc.)
        content: The content of the node
        spec_id: Optional SPEC node ID for alignment checking
        data: Optional metadata dictionary
        created_by: Who/what created this node
        check_alignment: Whether to verify alignment with spec
        signature: Optional AgentSignature for audit trail (recommended in production)

    Returns:
        SafeNodeResult with success status and validation details

    Note:
        If signature is provided, it will be stored in the node's metadata
        under the key '_signature_chain' as a SignatureChain object.
    """
    db = get_db()
    violations = []

    # Step 1: Syntax check for CODE nodes
    syntax_valid = True
    if node_type == NodeType.CODE.value:
        syntax_result = check_syntax(content, "python")
        syntax_valid = syntax_result.valid
        if not syntax_valid:
            violations.extend(syntax_result.errors)

    # If syntax fails, don't proceed
    if not syntax_valid:
        return SafeNodeResult(
            success=False,
            node_id="",
            syntax_valid=False,
            topology_valid=False,
            violations=violations,
            message=f"Syntax validation failed: {len(violations)} error(s)"
        )

    # Step 2: Create the node (but don't add yet for CODE with spec)
    try:
        # Prepare node data with signature chain if provided
        node_data = data or {}
        if signature is not None:
            # Import here to avoid circular dependency
            from agents.schemas import SignatureChain
            import uuid

            # Create a new signature chain for this node
            # Note: node_id will be set after NodeData.create()
            signature_chain = SignatureChain(
                node_id="",  # Will be updated after node creation
                state_id=str(uuid.uuid4()),
                signatures=[signature],
                is_replacement=False,
                replaced_node_id=None
            )
            # Store as dict for compatibility with msgspec
            node_data["_signature_chain"] = msgspec.structs.asdict(signature_chain)

        node = NodeData.create(
            type=node_type,
            content=content,
            data=node_data,
            created_by=created_by,
        )

        # Update signature chain with actual node_id if it was created
        if signature is not None and "_signature_chain" in node.data:
            node.data["_signature_chain"]["node_id"] = node.id

        # Step 3: Pre-flight topology check
        # For this, we'd need to temporarily add the node, which is complex
        # Instead, we validate after adding and rollback if needed
        db.add_node(node)

        # Step 4: Validate topology
        topology_violations = db.validate_topology(node.id, mode="hard")
        topology_valid = len(topology_violations) == 0
        violations.extend(topology_violations)

        # Step 5: Alignment check for CODE nodes with spec
        if node_type == NodeType.CODE.value and spec_id and check_alignment:
            # Add the IMPLEMENTS edge temporarily for alignment check
            try:
                edge = EdgeData.create(
                    source_id=spec_id,
                    target_id=node.id,
                    type=EdgeType.IMPLEMENTS.value,
                )
                db.add_edge(edge)

                # Now verify alignment
                alignment_result = verify_alignment(spec_id, node.id)
                if not alignment_result.approved:
                    violations.extend(alignment_result.violations)
                    # Rollback - remove node and edge
                    db.remove_edge(spec_id, node.id)
                    db.remove_node(node.id)

                    return SafeNodeResult(
                        success=False,
                        node_id="",
                        syntax_valid=syntax_valid,
                        topology_valid=topology_valid,
                        violations=violations,
                        message="Alignment verification failed"
                    )

            except Exception as e:
                violations.append(f"Alignment check error: {e}")

        # If we reach here with no critical violations, success!
        if violations and not topology_valid:
            # Rollback on topology violations
            db.remove_node(node.id)
            return SafeNodeResult(
                success=False,
                node_id="",
                syntax_valid=syntax_valid,
                topology_valid=False,
                violations=violations,
                message="Topology validation failed"
            )

        # Compute embedding for hybrid context assembly
        try:
            from core.embeddings import compute_embedding, is_available
            if is_available() and content:
                embedding = compute_embedding(content)
                if embedding is not None:
                    node.embedding = embedding
        except ImportError:
            pass  # Embeddings not available

        # Log successful node creation
        _log_node_created(node.id, node_type, created_by)

        # Log edge creation if IMPLEMENTS edge was added
        if node_type == NodeType.CODE.value and spec_id:
            _log_edge_created(spec_id, node.id, EdgeType.IMPLEMENTS.value)

        return SafeNodeResult(
            success=True,
            node_id=node.id,
            syntax_valid=syntax_valid,
            topology_valid=topology_valid,
            violations=violations,  # May have warnings
            message=f"Created and verified node {node.id}"
        )

    except DuplicateNodeError as e:
        return SafeNodeResult(
            success=False,
            node_id="",
            syntax_valid=syntax_valid,
            topology_valid=False,
            violations=[str(e)],
            message=str(e)
        )
    except Exception as e:
        return SafeNodeResult(
            success=False,
            node_id="",
            syntax_valid=syntax_valid,
            topology_valid=False,
            violations=[str(e)],
            message=f"Failed to create node: {e}"
        )


def update_node_status(
    node_id: Annotated[str, "Node ID to update"],
    new_status: Annotated[str, "New status (PENDING, PROCESSING, VERIFIED, FAILED, BLOCKED)"],
) -> ToolResult:
    """
    Update a node's status with validation.

    Args:
        node_id: The node to update
        new_status: The new status value

    Returns:
        ToolResult with success status
    """
    db = get_db()

    try:
        node = db.get_node(node_id)
        if node is None:
            return ToolResult(
                success=False,
                message=f"Node not found: {node_id}"
            )

        # Create updated node with new status
        updated = NodeData(
            id=node.id,
            type=node.type,
            content=node.content,
            status=new_status,
            data=node.data,
            created_by=node.created_by,
            created_at=node.created_at,
            version=node.version + 1,
        )

        db.update_node(node_id, updated)

        return ToolResult(
            success=True,
            message=f"Updated {node_id} status to {new_status}"
        )

    except Exception as e:
        return ToolResult(
            success=False,
            message=f"Failed to update status: {e}"
        )


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# All tools available to agents
TOOLS = {
    # Graph operations
    "add_node": add_node,
    "add_node_safe": add_node_safe,
    "add_nodes_batch": add_nodes_batch,
    "add_edge": add_edge,
    "add_edges_batch": add_edges_batch,
    "query_nodes": query_nodes,
    "get_node": get_node,
    "update_node_status": update_node_status,

    # Analysis
    "get_waves": get_waves,
    "get_descendants": get_descendants,
    "get_ancestors": get_ancestors,
    "check_cycle": check_cycle,
    "get_graph_stats": get_graph_stats,

    # Parsing
    "parse_source": parse_source,

    # Alignment
    "align_node_sets": align_node_sets,

    # Layer 7B - Auditor
    "check_syntax": check_syntax,
    "verify_alignment": verify_alignment,
}


def get_tool(name: str):
    """Get a tool by name."""
    return TOOLS.get(name)


def list_tools() -> List[str]:
    """List all available tool names."""
    return list(TOOLS.keys())
