"""
PARAGON API ROUTES - The HTTP Interface

RESTful API for ParagonDB operations using Starlette.

Endpoints:
- GET  /health           - Health check
- GET  /stats            - Graph statistics
- POST /nodes            - Add node(s)
- GET  /nodes/{id}       - Get node by ID
- GET  /nodes            - List nodes (with filters)
- POST /edges            - Add edge(s)
- GET  /edges            - List edges (with filters)
- GET  /waves            - Get wavefront layers
- POST /parse            - Parse source file/directory
- POST /align            - Align two graphs
- GET  /descendants/{id} - Get node descendants

Visualization Endpoints:
- GET  /api/viz/snapshot    - Get full graph snapshot (JSON)
- GET  /api/viz/stream      - Get graph as Arrow IPC stream
- GET  /api/viz/compare     - Compare two graph versions
- WS   /api/viz/ws          - WebSocket for real-time updates

Design:
- Starlette routes for ASGI compatibility with Granian
- msgspec for fast JSON serialization
- Arrow IPC for zero-copy graph transfer to Cosmograph
- All heavy operations delegated to core modules
"""
from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any, List, Set
import msgspec
import asyncio
from pathlib import Path
from datetime import datetime, timezone
import logging

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData, serialize_node, serialize_nodes, serialize_edges
from core.ontology import NodeType, NodeStatus, EdgeType
from domain.code_parser import CodeParser, parse_python_file, parse_python_directory
from viz.core import (
    GraphSnapshot,
    GraphDelta,
    VizNode,
    VizEdge,
    create_snapshot_from_db,
    serialize_to_arrow,
    compare_snapshots,
)
from agents.tools import get_db  # Use shared database instance

# Event bus for real-time graph change notifications
try:
    from infrastructure.event_bus import get_event_bus, GraphEvent, EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("paragon.api")


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Parser instance
_parser: Optional[CodeParser] = None
# NOTE: Database instance is now managed by agents.tools.get_db() for shared access

# WebSocket connections for real-time updates
_ws_connections: Set[WebSocket] = set()

# Snapshot cache for comparison view
_snapshot_cache: Dict[str, GraphSnapshot] = {}

# Sequence counter for delta messages
_sequence_counter: int = 0

# Orchestrator instance for dialectic interactions
_orchestrator_instance: Optional[Any] = None
_orchestrator_state: Dict[str, Any] = {}


def _load_session_state() -> Dict[str, Any]:
    """Load session state from file (written by main.py)."""
    import json
    state_file = Path(__file__).parent.parent / "workspace" / "current_session.json"

    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load session state: {e}")
    return {}


def _ensure_session_state_loaded():
    """Ensure session state is loaded from file."""
    global _orchestrator_state
    if not _orchestrator_state or not _orchestrator_state.get("session_id"):
        loaded_state = _load_session_state()
        if loaded_state:
            _orchestrator_state.update(loaded_state)
            print(f"Loaded session state: session_id={_orchestrator_state.get('session_id')}, title={_orchestrator_state.get('title')}")


def _refresh_session_state():
    """Refresh session state from file to pick up orchestrator updates.

    This should be called on each API request to get the latest state,
    since the orchestrator thread updates the file when questions are generated.
    """
    global _orchestrator_state
    loaded_state = _load_session_state()
    if loaded_state:
        _orchestrator_state.update(loaded_state)


def _save_session_state():
    """Save session state to file to persist API updates.

    This should be called when the API updates state (e.g., after approval)
    so the orchestrator thread can pick up the changes.
    """
    import json
    from datetime import datetime

    global _orchestrator_state
    state_file = Path(__file__).parent.parent / "workspace" / "current_session.json"

    try:
        # Update timestamp
        _orchestrator_state["updated_at"] = datetime.utcnow().isoformat() + "Z"

        with open(state_file, "w") as f:
            json.dump(_orchestrator_state, f, indent=2)
        print(f"[API] Saved session state: phase={_orchestrator_state.get('phase')}")
    except Exception as e:
        print(f"Warning: Could not save session state: {e}")


def _get_current_phase_from_diagnostics(session_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the current phase from diagnostics log for a session.

    Returns the most recent phase entry and any recent LLM calls.
    """
    import json
    log_file = Path(__file__).parent.parent / "workspace" / "logs" / "diagnostics.jsonl"

    if not log_file.exists():
        return None

    # Read the last 50 lines (efficient for JSONL)
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()[-50:]

        current_phase = None
        last_llm_call = None
        llm_call_count = 0

        for line in lines:
            try:
                entry = json.loads(line.strip())
                entry_session = entry.get("session_id", "")

                # Match session ID (may have suffix like _77478f4e)
                if session_id and entry_session and (
                    entry_session.startswith(session_id) or
                    session_id in entry_session
                ):
                    if entry.get("type") == "phase":
                        current_phase = entry.get("phase")
                    elif entry.get("type") == "llm_call":
                        last_llm_call = entry.get("schema")
                        llm_call_count += 1
            except json.JSONDecodeError:
                continue

        return {
            "phase": current_phase,
            "last_llm_call": last_llm_call,
            "llm_call_count": llm_call_count,
        }
    except Exception as e:
        print(f"Warning: Could not read diagnostics: {e}")
        return None


def _next_sequence() -> int:
    """Get next sequence number for delta messages."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


def get_parser() -> CodeParser:
    """Get the global parser instance."""
    global _parser
    if _parser is None:
        _parser = CodeParser()
    return _parser


def get_orchestrator():
    """Get the global orchestrator instance (lazy loaded)."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        try:
            from agents.orchestrator import TDDOrchestrator
            _orchestrator_instance = TDDOrchestrator(enable_checkpointing=True)
        except ImportError:
            # Graceful degradation if orchestrator not available
            _orchestrator_instance = None
    return _orchestrator_instance


# =============================================================================
# RESPONSE HELPERS
# =============================================================================

# Pre-compiled msgspec encoder for fast JSON serialization
_json_encoder = msgspec.json.Encoder()


def json_response(data: Any, status_code: int = 200) -> Response:
    """Create JSON response using msgspec for speed."""
    if isinstance(data, (NodeData, EdgeData)):
        # Use msgspec for schema types
        body = _json_encoder.encode(data)
    elif isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], NodeData):
            body = serialize_nodes(data)
        elif isinstance(data[0], EdgeData):
            body = serialize_edges(data)
        else:
            body = _json_encoder.encode(data)
    else:
        body = _json_encoder.encode(data)

    return Response(
        content=body,
        status_code=status_code,
        media_type="application/json"
    )


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Create error response."""
    return JSONResponse(
        {"error": message},
        status_code=status_code
    )


# =============================================================================
# HEALTH & STATS
# =============================================================================

async def health(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "paragon",
        "version": "0.1.0"
    })


async def stats(request: Request) -> JSONResponse:
    """Get graph statistics."""
    db = get_db()
    return JSONResponse({
        "node_count": db.node_count,
        "edge_count": db.edge_count,
        "has_cycle": db.has_cycle(),
        "is_empty": db.is_empty,
    })


# =============================================================================
# NODE OPERATIONS
# =============================================================================

async def create_node(request: Request) -> Response:
    """
    Create one or more nodes.

    Body (single):
        {"type": "CODE", "content": "...", "data": {...}}

    Body (batch):
        [{"type": "CODE", ...}, {"type": "SPEC", ...}]
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    db = get_db()

    # Validate and sanitize content
    def validate_node_data(item: dict, index: int = 0) -> tuple[bool, str]:
        content = item.get("content", "").strip() if isinstance(item.get("content"), str) else ""
        node_type = item.get("type", "").strip() if isinstance(item.get("type"), str) else ""

        if not content:
            return False, f"Node {index}: content is required and cannot be empty"

        if len(content) > 100000:
            return False, f"Node {index}: content too long (max 100000 characters)"

        if node_type and len(node_type) > 50:
            return False, f"Node {index}: type too long (max 50 characters)"

        created_by = item.get("created_by", "").strip() if isinstance(item.get("created_by"), str) else ""
        if created_by and len(created_by) > 100:
            return False, f"Node {index}: created_by too long (max 100 characters)"

        # Update item with sanitized values
        item["content"] = content
        if node_type:
            item["type"] = node_type
        if created_by:
            item["created_by"] = created_by

        return True, ""

    # Handle batch vs single
    if isinstance(body, list):
        if len(body) > 1000:
            return error_response("Batch too large (max 1000 nodes)", status_code=400)

        nodes = []
        for idx, item in enumerate(body):
            valid, error_msg = validate_node_data(item, idx)
            if not valid:
                return error_response(error_msg, status_code=400)

            node = NodeData.create(
                type=item.get("type", NodeType.CODE.value),
                content=item.get("content", ""),
                data=item.get("data", {}),
                created_by=item.get("created_by", "api"),
            )
            nodes.append(node)
        db.add_nodes_batch(nodes)

        # Broadcast delta to WebSocket clients
        if _ws_connections:
            viz_nodes = [VizNode.from_node_data(n) for n in nodes]
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_added=viz_nodes,
            )
            await broadcast_delta(delta)

        return json_response({"created": len(nodes), "ids": [n.id for n in nodes]}, status_code=201)
    else:
        valid, error_msg = validate_node_data(body)
        if not valid:
            return error_response(error_msg, status_code=400)

        node = NodeData.create(
            type=body.get("type", NodeType.CODE.value),
            content=body.get("content", ""),
            data=body.get("data", {}),
            created_by=body.get("created_by", "api"),
        )
        db.add_node(node)

        # Broadcast delta to WebSocket clients
        if _ws_connections:
            viz_node = VizNode.from_node_data(node)
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_added=[viz_node],
            )
            await broadcast_delta(delta)

        return json_response({"id": node.id}, status_code=201)


async def get_node(request: Request) -> Response:
    """Get node by ID."""
    node_id = request.path_params["node_id"]
    db = get_db()

    node = db.get_node(node_id)
    if node is None:
        return error_response(f"Node not found: {node_id}", status_code=404)

    return json_response(node)


async def get_node_dialogue(request: Request) -> JSONResponse:
    """
    Get all dialogue turns linked to a node.

    This traverses the graph to find CLARIFICATION nodes connected
    to this node or its ancestors, returning Q&A pairs as dialogue turns.

    Returns:
        {
            "node_id": "...",
            "dialogue": [
                {
                    "turn_number": 0,
                    "agent": "system",
                    "type": "question",
                    "content": "What does 'fast' mean?",
                    "timestamp": "...",
                    "metadata": {...}
                },
                {
                    "turn_number": 0,
                    "agent": "user",
                    "type": "answer",
                    "content": "< 100ms latency",
                    "timestamp": "...",
                    "metadata": {...}
                }
            ],
            "count": 2
        }
    """
    node_id = request.path_params["node_id"]
    db = get_db()

    node = db.get_node(node_id)
    if node is None:
        return error_response(f"Node not found: {node_id}", status_code=404)

    try:
        # Strategy: Find CLARIFICATION nodes in ancestors
        # 1. Get ancestors of this node
        ancestors = db.get_ancestors(node_id)
        ancestor_ids = {a.id for a in ancestors}
        ancestor_ids.add(node_id)  # Include self

        # 2. Find all CLARIFICATION nodes
        all_nodes = db.get_all_nodes()
        clarification_nodes = [n for n in all_nodes if n.type == "CLARIFICATION"]

        # 3. Find CLARIFICATION nodes that TRACE_TO any of our ancestors
        edges = db.get_all_edges()
        relevant_clarifications = []

        for clarif_node in clarification_nodes:
            # Check if this CLARIFICATION traces to any of our ancestors
            for edge in edges:
                if (edge.type == "TRACES_TO" and
                    edge.source_id == clarif_node.id and
                    edge.target_id in ancestor_ids):
                    relevant_clarifications.append(clarif_node)
                    break

        # 4. Build dialogue turns from CLARIFICATION nodes
        # Group by question-answer pairs using RESOLVED_BY edges
        dialogue_turns = []
        processed = set()

        for node in relevant_clarifications:
            if node.id in processed:
                continue

            role = node.data.get("role", "unknown")

            if role == "question":
                # This is a question - find its answer
                question_turn = {
                    "turn_number": node.data.get("turn_number", 0),
                    "agent": "system",
                    "type": "question",
                    "content": node.content,
                    "timestamp": node.data.get("timestamp", node.created_at),
                    "metadata": {
                        "node_id": node.id,
                        "category": node.data.get("category", ""),
                        "session_id": node.data.get("session_id", ""),
                    }
                }
                dialogue_turns.append(question_turn)
                processed.add(node.id)

                # Find answer via RESOLVED_BY edge
                for edge in edges:
                    if edge.type == "RESOLVED_BY" and edge.target_id == node.id:
                        answer_node = db.get_node(edge.source_id)
                        if answer_node:
                            answer_turn = {
                                "turn_number": answer_node.data.get("turn_number", 0),
                                "agent": "user",
                                "type": "answer",
                                "content": answer_node.content,
                                "timestamp": answer_node.data.get("timestamp", answer_node.created_at),
                                "metadata": {
                                    "node_id": answer_node.id,
                                    "category": answer_node.data.get("category", ""),
                                    "session_id": answer_node.data.get("session_id", ""),
                                }
                            }
                            dialogue_turns.append(answer_turn)
                            processed.add(answer_node.id)
                        break

        # Sort by turn_number and timestamp
        dialogue_turns.sort(key=lambda t: (t["turn_number"], t["timestamp"]))

        return JSONResponse({
            "node_id": node_id,
            "dialogue": dialogue_turns,
            "count": len(dialogue_turns),
        })

    except Exception as e:
        return error_response(f"Failed to retrieve dialogue: {e}", status_code=500)


async def get_research_nodes(request: Request) -> JSONResponse:
    """
    Get all RESEARCH nodes with full data including findings and search results.

    Returns:
        [
            {
                "node_id": "...",
                "req_node_id": "...",
                "iteration": 1,
                "query": "...",
                "total_findings": 5,
                "total_ambiguities": 2,
                "blocking_count": 0,
                "out_of_scope": [],
                "synthesis": "...",
                "findings": [...],
                "ambiguities": [...],
                "search_results": [...],
                "created_at": "...",
                "status": "..."
            }
        ]
    """
    db = get_db()

    # Get all RESEARCH nodes
    all_nodes = db.get_all_nodes()
    research_nodes = [n for n in all_nodes if n.type == NodeType.RESEARCH.value]

    # Get all edges to find REQ relationships
    all_edges = db.get_all_edges()

    # Build response with detailed research data
    research_data = []
    for node in research_nodes:
        # Find the REQ node this research is for
        req_node_id = ""
        for edge in all_edges:
            if (edge.type == EdgeType.RESEARCH_FOR.value and
                edge.source_id == node.id):
                req_node_id = edge.target_id
                break

        # Extract data from node
        data = node.data if node.data else {}

        research_item = {
            "node_id": node.id,
            "req_node_id": req_node_id,
            "iteration": data.get("iteration", 0),
            "query": data.get("query", ""),
            "total_findings": data.get("total_findings", 0),
            "total_ambiguities": data.get("total_ambiguities", 0),
            "blocking_count": data.get("blocking_count", 0),
            "out_of_scope": data.get("out_of_scope", []),
            "synthesis": node.content,
            "findings": data.get("findings", []),
            "ambiguities": data.get("ambiguities", []),
            "search_results": data.get("search_results", []),
            "created_at": node.created_at,
            "status": node.status,
        }

        research_data.append(research_item)

    # Sort by created_at descending (newest first)
    research_data.sort(key=lambda x: x["created_at"], reverse=True)

    return JSONResponse({"research_nodes": research_data, "count": len(research_data)})


async def list_nodes(request: Request) -> Response:
    """
    List nodes with optional filters.

    Query params:
        type: Filter by node type
        status: Filter by status
        limit: Max results (default 100)
    """
    db = get_db()

    # Get query params
    node_type = request.query_params.get("type")
    status = request.query_params.get("status")
    limit = int(request.query_params.get("limit", 100))

    # Get all nodes and filter
    nodes = db.get_all_nodes()

    if node_type:
        nodes = [n for n in nodes if n.type == node_type]
    if status:
        nodes = [n for n in nodes if n.status == status]

    # Apply limit
    nodes = nodes[:limit]

    return json_response(nodes)


# =============================================================================
# EDGE OPERATIONS
# =============================================================================

async def create_edge(request: Request) -> Response:
    """
    Create one or more edges.

    Body (single):
        {"source_id": "...", "target_id": "...", "type": "DEPENDS_ON"}

    Body (batch):
        [{"source_id": "...", "target_id": "...", "type": "..."}, ...]
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    db = get_db()

    if isinstance(body, list):
        edges = []
        for item in body:
            edge = EdgeData.create(
                source_id=item["source_id"],
                target_id=item["target_id"],
                type=item.get("type", EdgeType.DEPENDS_ON.value),
                weight=item.get("weight", 1.0),
                metadata=item.get("metadata", {}),
            )
            edges.append(edge)
        db.add_edges_batch(edges)

        # Broadcast delta to WebSocket clients
        if _ws_connections:
            viz_edges = [VizEdge.from_edge_data(e) for e in edges]
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                edges_added=viz_edges,
            )
            await broadcast_delta(delta)

        return json_response({"created": len(edges)}, status_code=201)
    else:
        edge = EdgeData.create(
            source_id=body["source_id"],
            target_id=body["target_id"],
            type=body.get("type", EdgeType.DEPENDS_ON.value),
            weight=body.get("weight", 1.0),
            metadata=body.get("metadata", {}),
        )
        db.add_edge(edge)

        # Broadcast delta to WebSocket clients
        if _ws_connections:
            viz_edge = VizEdge.from_edge_data(edge)
            delta = GraphDelta(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sequence=_next_sequence(),
                edges_added=[viz_edge],
            )
            await broadcast_delta(delta)

        return json_response({"created": 1}, status_code=201)


async def list_edges(request: Request) -> Response:
    """
    List edges with optional filters.

    Query params:
        type: Filter by edge type
        source_id: Filter by source node
        target_id: Filter by target node
        limit: Max results (default 100)
    """
    db = get_db()

    edge_type = request.query_params.get("type")
    source_id = request.query_params.get("source_id")
    target_id = request.query_params.get("target_id")
    limit = int(request.query_params.get("limit", 100))

    edges = db.get_all_edges()

    if edge_type:
        edges = [e for e in edges if e.type == edge_type]
    if source_id:
        edges = [e for e in edges if e.source_id == source_id]
    if target_id:
        edges = [e for e in edges if e.target_id == target_id]

    edges = edges[:limit]

    return json_response(edges)


# =============================================================================
# GRAPH OPERATIONS
# =============================================================================

async def get_waves(request: Request) -> JSONResponse:
    """
    Get wavefront layers (topological sort into layers).

    Returns layers where each layer contains node IDs that can be
    processed in parallel.
    """
    db = get_db()
    waves = db.get_waves()
    # Convert NodeData to just IDs for JSON serialization
    wave_ids = [[node.id for node in layer] for layer in waves]
    return JSONResponse({
        "layer_count": len(wave_ids),
        "layers": wave_ids
    })


async def get_descendants(request: Request) -> JSONResponse:
    """Get all descendants of a node."""
    node_id = request.path_params["node_id"]
    db = get_db()

    try:
        descendants = db.get_descendants(node_id)
        return JSONResponse({
            "node_id": node_id,
            "descendant_count": len(descendants),
            "descendants": descendants
        })
    except KeyError:
        return error_response(f"Node not found: {node_id}", status_code=404)


async def get_ancestors(request: Request) -> JSONResponse:
    """Get all ancestors of a node."""
    node_id = request.path_params["node_id"]
    db = get_db()

    try:
        ancestors = db.get_ancestors(node_id)
        return JSONResponse({
            "node_id": node_id,
            "ancestor_count": len(ancestors),
            "ancestors": ancestors
        })
    except KeyError:
        return error_response(f"Node not found: {node_id}", status_code=404)


# =============================================================================
# PARSING OPERATIONS
# =============================================================================

async def parse_source(request: Request) -> JSONResponse:
    """
    Parse source file or directory.

    Body:
        {"path": "/path/to/file.py"} or {"path": "/path/to/dir", "recursive": true}
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    path = Path(body.get("path", ""))
    if not path.exists():
        return error_response(f"Path not found: {path}", status_code=404)

    db = get_db()
    parser = get_parser()

    if path.is_file():
        nodes, edges = parser.parse_file(path)
    else:
        recursive = body.get("recursive", True)
        nodes, edges = parse_python_directory(path, recursive=recursive)

    # Add to database
    db.add_nodes_batch(nodes)

    # Only add edges where both nodes exist
    valid_node_ids = {n.id for n in nodes}
    valid_edges = [
        e for e in edges
        if e.source_id in valid_node_ids and e.target_id in valid_node_ids
    ]
    db.add_edges_batch(valid_edges)

    return JSONResponse({
        "nodes_added": len(nodes),
        "edges_added": len(valid_edges),
        "path": str(path),
    })


# =============================================================================
# ALIGNMENT OPERATIONS
# =============================================================================

async def align_graphs(request: Request) -> JSONResponse:
    """
    Align two sets of nodes.

    Body:
        {
            "source_ids": ["id1", "id2", ...],
            "target_ids": ["id3", "id4", ...],
            "algorithm": "rrwm"  # optional: rrwm, ipfp, sm, hungarian
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    from core.alignment import GraphAligner, MatchingAlgorithm

    db = get_db()

    source_ids = body.get("source_ids", [])
    target_ids = body.get("target_ids", [])
    algorithm = body.get("algorithm", "rrwm")

    # Get nodes
    source_nodes = [db.get_node(nid) for nid in source_ids if db.get_node(nid)]
    target_nodes = [db.get_node(nid) for nid in target_ids if db.get_node(nid)]

    if not source_nodes or not target_nodes:
        return error_response("No valid nodes found")

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
    try:
        algo = MatchingAlgorithm(algorithm)
    except ValueError:
        return error_response(f"Unknown algorithm: {algorithm}")

    aligner = GraphAligner(algorithm=algo)
    result = aligner.align(source_nodes, source_edges, target_nodes, target_edges)

    return JSONResponse({
        "score": result.score,
        "mappings": result.node_mapping,
        "unmapped_source": result.unmapped_source,
        "unmapped_target": result.unmapped_target,
    })


# =============================================================================
# VISUALIZATION ENDPOINTS
# =============================================================================

async def viz_snapshot(request: Request) -> JSONResponse:
    """
    Get a complete graph snapshot for visualization.

    Query params:
        color_mode: "type" (default) or "status"
        version: Optional version label for caching
    """
    db = get_db()

    color_mode = request.query_params.get("color_mode", "type")
    version = request.query_params.get("version", "current")

    snapshot = create_snapshot_from_db(db, color_mode=color_mode, version=version)

    # Cache for comparison
    _snapshot_cache[version] = snapshot

    return JSONResponse(snapshot.to_dict())


async def viz_stream(request: Request) -> Response:
    """
    Get graph as Apache Arrow IPC stream.

    This is the high-performance endpoint for Cosmograph.
    Returns two Arrow IPC files concatenated with a length prefix.

    Query params:
        color_mode: "type" (default) or "status"
        format: "nodes" | "edges" | "both" (default: "both")
    """
    db = get_db()

    color_mode = request.query_params.get("color_mode", "type")
    format_type = request.query_params.get("format", "both")

    snapshot = create_snapshot_from_db(db, color_mode=color_mode)
    nodes_bytes, edges_bytes = serialize_to_arrow(snapshot)

    if format_type == "nodes":
        return Response(
            content=nodes_bytes,
            media_type="application/vnd.apache.arrow.stream",
            headers={"Content-Disposition": "attachment; filename=nodes.arrow"}
        )
    elif format_type == "edges":
        return Response(
            content=edges_bytes,
            media_type="application/vnd.apache.arrow.stream",
            headers={"Content-Disposition": "attachment; filename=edges.arrow"}
        )
    else:
        # Return both with length prefix
        import struct
        combined = struct.pack("<I", len(nodes_bytes)) + nodes_bytes + edges_bytes
        return Response(
            content=combined,
            media_type="application/vnd.apache.arrow.stream",
            headers={"Content-Disposition": "attachment; filename=graph.arrow"}
        )


async def viz_compare(request: Request) -> JSONResponse:
    """
    Compare two graph snapshots.

    Used for the Development View regression testing.

    Query params:
        baseline: Version of baseline snapshot (must be cached)
        treatment: Version of treatment snapshot (default: "current")
    """
    db = get_db()

    baseline_version = request.query_params.get("baseline")
    treatment_version = request.query_params.get("treatment", "current")

    # Get baseline from cache
    if baseline_version and baseline_version in _snapshot_cache:
        baseline = _snapshot_cache[baseline_version]
    else:
        return error_response(
            f"Baseline version '{baseline_version}' not found. "
            "First call /api/viz/snapshot?version=<name> to create it.",
            status_code=400
        )

    # Get or create treatment
    if treatment_version in _snapshot_cache:
        treatment = _snapshot_cache[treatment_version]
    else:
        treatment = create_snapshot_from_db(db, version=treatment_version)
        _snapshot_cache[treatment_version] = treatment

    comparison = compare_snapshots(baseline, treatment)
    return JSONResponse(comparison.to_dict())


async def viz_save_snapshot(request: Request) -> JSONResponse:
    """
    Save current graph state as a named snapshot.

    Body:
        {"version": "v1.0", "label": "Before refactoring"}
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    db = get_db()
    version = body.get("version", "snapshot")
    label = body.get("label", "")

    snapshot = create_snapshot_from_db(db, version=version, label=label)
    _snapshot_cache[version] = snapshot

    return JSONResponse({
        "saved": True,
        "version": version,
        "node_count": snapshot.node_count,
        "edge_count": snapshot.edge_count,
    })


async def viz_list_snapshots(request: Request) -> JSONResponse:
    """List all cached snapshots."""
    snapshots = []
    for version, snapshot in _snapshot_cache.items():
        snapshots.append({
            "version": version,
            "label": snapshot.label,
            "node_count": snapshot.node_count,
            "edge_count": snapshot.edge_count,
            "timestamp": snapshot.timestamp,
        })
    return JSONResponse({"snapshots": snapshots})


async def viz_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time graph updates.

    Protocol:
    1. Client connects
    2. Server sends initial snapshot
    3. Server sends deltas on graph changes
    4. Client can send commands (color_mode change, etc.)
    """
    await websocket.accept()
    _ws_connections.add(websocket)

    try:
        # Send initial snapshot
        db = get_db()
        snapshot = create_snapshot_from_db(db)
        await websocket.send_json({
            "type": "snapshot",
            "data": snapshot.to_dict()
        })

        # Listen for client commands and keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0  # 30s heartbeat
                )

                # Handle client commands
                if data.get("type") == "color_mode":
                    color_mode = data.get("mode", "type")
                    snapshot = create_snapshot_from_db(db, color_mode=color_mode)
                    await websocket.send_json({
                        "type": "snapshot",
                        "data": snapshot.to_dict()
                    })
                elif data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    finally:
        _ws_connections.discard(websocket)


async def broadcast_delta(delta: GraphDelta) -> None:
    """
    Broadcast a graph delta to all connected WebSocket clients.

    Called by node/edge mutation handlers.
    """
    if not _ws_connections:
        return

    message = {
        "type": "delta",
        "data": delta.to_dict()
    }

    # Send to all connections, removing dead ones
    dead = set()
    for ws in _ws_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)

    _ws_connections.difference_update(dead)


async def broadcast_json(message: Dict[str, Any]) -> None:
    """
    Broadcast a JSON message to all connected WebSocket clients.

    Args:
        message: Dictionary to send as JSON
    """
    if not _ws_connections:
        return

    # Send to all connections, removing dead ones
    dead = set()
    for ws in _ws_connections:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)

    _ws_connections.difference_update(dead)


async def broadcast_graph_event(event: GraphEvent) -> None:
    """
    Convert graph events to WebSocket deltas and broadcast.

    Called by event bus whenever graph changes occur.

    This is the bridge between the event bus (decoupled) and the WebSocket
    layer (presentation). It converts internal GraphEvent objects to
    VizNode/VizEdge deltas for frontend consumption.

    Args:
        event: GraphEvent from event bus
    """
    if not _ws_connections:
        return

    db = get_db()

    try:
        if event.type == EventType.NODE_CREATED:
            # Fetch full node data from DB
            node_id = event.payload.get("node_id")
            if not node_id:
                return

            try:
                node_data = db.get_node(node_id)
                viz_node = VizNode.from_node_data(node_data)

                delta = GraphDelta(
                    timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                    sequence=_next_sequence(),
                    nodes_added=[viz_node],
                    nodes_updated=[],
                    nodes_removed=[],
                    edges_added=[],
                    edges_removed=[],
                )
                await broadcast_delta(delta)

            except Exception as e:
                logger.error(f"Failed to broadcast node creation: {e}")

        elif event.type == EventType.NODE_UPDATED:
            # Fetch updated node data
            node_id = event.payload.get("node_id")
            if not node_id:
                return

            try:
                node_data = db.get_node(node_id)
                viz_node = VizNode.from_node_data(node_data)

                delta = GraphDelta(
                    timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                    sequence=_next_sequence(),
                    nodes_added=[],
                    nodes_updated=[viz_node],
                    nodes_removed=[],
                    edges_added=[],
                    edges_removed=[],
                )
                await broadcast_delta(delta)

            except Exception as e:
                logger.error(f"Failed to broadcast node update: {e}")

        elif event.type == EventType.NODE_DELETED:
            # Send removal delta
            node_id = event.payload.get("node_id")
            if not node_id:
                return

            delta = GraphDelta(
                timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_added=[],
                nodes_updated=[],
                nodes_removed=[node_id],
                edges_added=[],
                edges_removed=[],
            )
            await broadcast_delta(delta)

        elif event.type == EventType.EDGE_CREATED:
            # Fetch full edge data
            source_id = event.payload.get("source_id")
            target_id = event.payload.get("target_id")
            if not source_id or not target_id:
                return

            try:
                edge_data = db.get_edge(source_id, target_id)
                viz_edge = VizEdge.from_edge_data(edge_data)

                delta = GraphDelta(
                    timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                    sequence=_next_sequence(),
                    nodes_added=[],
                    nodes_updated=[],
                    nodes_removed=[],
                    edges_added=[viz_edge],
                    edges_removed=[],
                )
                await broadcast_delta(delta)

            except Exception as e:
                logger.error(f"Failed to broadcast edge creation: {e}")

        elif event.type == EventType.EDGE_DELETED:
            # Send edge removal delta
            source_id = event.payload.get("source_id")
            target_id = event.payload.get("target_id")
            if not source_id or not target_id:
                return

            delta = GraphDelta(
                timestamp=datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                sequence=_next_sequence(),
                nodes_added=[],
                nodes_updated=[],
                nodes_removed=[],
                edges_added=[],
                edges_removed=[(source_id, target_id)],
            )
            await broadcast_delta(delta)

        elif event.type == EventType.ORCHESTRATOR_ERROR:
            # Broadcast error message
            error_msg = {
                "type": "error",
                "error": event.payload.get("error_message", "Unknown error"),
                "phase": event.payload.get("phase", "UNKNOWN"),
                "error_type": event.payload.get("error_type", "Exception"),
                "timestamp": datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                "source": event.source,
            }
            await broadcast_json(error_msg)

        elif event.type == EventType.PHASE_CHANGED:
            # Broadcast phase change (for dialectic WebSocket)
            phase = event.payload.get("phase", "UNKNOWN")
            session_id = event.payload.get("session_id", "")

            if session_id:
                await broadcast_phase_change(session_id, phase)

        elif event.type == EventType.NOTIFICATION_CREATED:
            # Broadcast notification to connected clients
            notification_msg = {
                "type": "notification",
                "data": {
                    "notification_id": event.payload.get("metadata", {}).get("notification_id", ""),
                    "type": event.payload.get("notification_type", "info"),
                    "message": event.payload.get("message", ""),
                    "target_tabs": event.payload.get("target_tabs", []),
                    "urgency": event.payload.get("urgency", "info"),
                    "metadata": event.payload.get("metadata", {}),
                    "related_node_id": event.payload.get("related_node_id"),
                    "timestamp": datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                    "source": event.source,
                }
            }
            await broadcast_json(notification_msg)

        elif event.type == EventType.DIALOGUE_TURN_ADDED:
            # Broadcast dialogue turn update
            dialogue_msg = {
                "type": "dialogue_turn",
                "data": {
                    "node_id": event.payload.get("node_id"),
                    "turn_number": event.payload.get("turn_number"),
                    "agent": event.payload.get("agent"),
                    "turn_type": event.payload.get("type"),
                    "content": event.payload.get("content"),
                    "metadata": event.payload.get("metadata", {}),
                    "timestamp": datetime.fromtimestamp(event.timestamp, timezone.utc).isoformat(),
                }
            }
            await broadcast_json(dialogue_msg)

    except Exception as e:
        logger.error(f"Failed to broadcast graph event {event.type.value}: {e}")


async def broadcast_node_highlight(
    nodes_to_highlight: List[str],
    edges_to_highlight: List[Dict[str, str]],
    reason: str,
    highlight_mode: str = "related"
) -> None:
    """
    Broadcast a node highlight event to all connected WebSocket clients.

    This enables real-time highlighting updates when nodes/messages are
    clicked in the UI.

    Args:
        nodes_to_highlight: List of node IDs to highlight
        edges_to_highlight: List of edge dicts with 'source' and 'target'
        reason: Human-readable reason for highlighting
        highlight_mode: Mode used (exact, related, dependent)

    Example:
        await broadcast_node_highlight(
            nodes_to_highlight=["node_1", "node_2"],
            edges_to_highlight=[{"source": "node_1", "target": "node_2"}],
            reason="User clicked message #5",
            highlight_mode="related"
        )
    """
    if not _ws_connections:
        return

    message = {
        "type": "node_highlight",
        "data": {
            "nodes_to_highlight": nodes_to_highlight,
            "edges_to_highlight": edges_to_highlight,
            "reason": reason,
            "highlight_mode": highlight_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }

    await broadcast_json(message)


# =============================================================================
# DIALECTIC ENDPOINTS (Phase 3 - Human-in-the-Loop)
# =============================================================================

async def get_dialector_questions(request: Request) -> JSONResponse:
    """
    Get current ambiguity questions from the orchestrator.

    Returns pending clarification questions from the dialectic phase.

    Response:
        {
            "questions": [
                {
                    "id": "question_id",
                    "text": "What specific criteria define 'fast'?",
                    "category": "SUBJECTIVE_TERMS",
                    "suggested_answer": "< 100ms latency"
                }
            ],
            "phase": "CLARIFICATION",
            "session_id": "session_id"
        }
    """
    global _orchestrator_state

    # Check if we have an active orchestrator state with pending questions
    questions = _orchestrator_state.get("clarification_questions", [])
    phase = _orchestrator_state.get("phase", "INIT")
    session_id = _orchestrator_state.get("session_id", "")

    # Format questions for frontend
    formatted_questions = []
    for i, q in enumerate(questions):
        formatted_questions.append({
            "id": f"q_{i}",
            "text": q.get("question", ""),
            "category": q.get("category", ""),
            "suggested_answer": q.get("suggested_answer", ""),
            "ambiguity_text": q.get("text", ""),
        })

    return JSONResponse({
        "questions": formatted_questions,
        "phase": phase,
        "session_id": session_id,
        "has_questions": len(formatted_questions) > 0,
    })


async def submit_dialector_answer(request: Request) -> JSONResponse:
    """
    Submit user answers to ambiguity questions.

    Body:
        {
            "session_id": "session_id",
            "answers": [
                {"question_id": "q_0", "answer": "Response latency < 100ms"},
                {"question_id": "q_1", "answer": "REST API"}
            ]
        }

    Response:
        {
            "success": true,
            "message": "Answers submitted, orchestrator resumed",
            "new_phase": "research"
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    session_id = body.get("session_id", "").strip()
    answers = body.get("answers", [])

    # Validate session_id
    if not session_id:
        return error_response("session_id is required and cannot be empty", status_code=400)

    if len(session_id) > 100:
        return error_response("session_id too long (max 100 characters)", status_code=400)

    # Validate answers
    if not answers:
        return error_response("answers array is required and cannot be empty", status_code=400)

    if not isinstance(answers, list):
        return error_response("answers must be an array", status_code=400)

    # Validate each answer
    for i, ans in enumerate(answers):
        if not isinstance(ans, dict):
            return error_response(f"Answer {i} must be an object", status_code=400)

        answer_text = ans.get("answer", "").strip()
        if not answer_text:
            return error_response(f"Answer {i} cannot be empty", status_code=400)

        if len(answer_text) > 2000:
            return error_response(f"Answer {i} too long (max 2000 characters)", status_code=400)

        # Update with sanitized answer
        ans["answer"] = answer_text

    # Check if this is a Socratic approval (special case)
    is_socratic_approval = any(
        ans.get("question_id") == "socratic_approval"
        for ans in answers
    )

    # Format answers as a human response
    answer_text = "\n".join([
        f"{i+1}. {ans.get('answer', '')}"
        for i, ans in enumerate(answers)
    ])

    global _orchestrator_state
    _orchestrator_state["human_response"] = answer_text
    _orchestrator_state["pending_human_input"] = None

    # Handle Socratic approval specially
    if is_socratic_approval:
        # Check if this is an approval (contains "Approved" or "approve") or rejection
        first_answer = answers[0].get("answer", "").lower()
        is_approved = "approved" in first_answer or "proceed" in first_answer

        if is_approved:
            _orchestrator_state["phase"] = "PLAN"
            new_phase = "PLAN"
            print(f"[API] Socratic approval received. Transitioning to PLAN phase.")
        else:
            # Request for revisions - stay in clarification, generate new questions
            _orchestrator_state["phase"] = "CLARIFICATION"
            new_phase = "CLARIFICATION"
            print(f"[API] Revision requested. Staying in CLARIFICATION phase.")

        # Persist state to file so orchestrator thread can pick it up
        _save_session_state()

        return JSONResponse({
            "success": True,
            "message": "Approval submitted successfully" if is_approved else "Revision request submitted",
            "new_phase": new_phase,
            "session_id": session_id,
        })

    # Try to resume orchestrator if available (for regular Q&A)
    orchestrator = get_orchestrator()
    new_phase = "research"  # Default next phase after clarification

    if orchestrator and session_id:
        try:
            # Resume the orchestrator with the human response
            result = orchestrator.resume(session_id, human_response=answer_text)
            new_phase = result.get("phase", "research")
            _orchestrator_state.update(result)
        except Exception as e:
            # If resume fails, just update state manually
            _orchestrator_state["phase"] = "research"
            new_phase = "research"

    # Persist state for regular answers too
    _save_session_state()

    return JSONResponse({
        "success": True,
        "message": "Answers submitted successfully",
        "new_phase": new_phase,
        "session_id": session_id,
    })


async def get_orchestrator_state(request: Request) -> JSONResponse:
    """
    Get current orchestrator state.

    Query params:
        session_id: Optional session filter

    Response:
        {
            "phase": "CLARIFICATION",
            "session_id": "...",
            "has_pending_input": true,
            "iteration": 0
        }
    """
    session_id = request.query_params.get("session_id")

    orchestrator = get_orchestrator()
    if orchestrator and session_id:
        try:
            state = orchestrator.get_state(session_id)
            if state:
                return JSONResponse({
                    "phase": state.get("phase", "INIT"),
                    "session_id": session_id,
                    "has_pending_input": state.get("pending_human_input") is not None,
                    "iteration": state.get("iteration", 0),
                    "research_complete": state.get("research_complete", False),
                })
        except Exception:
            pass

    # Fallback to global state
    global _orchestrator_state
    return JSONResponse({
        "phase": _orchestrator_state.get("phase", "init"),
        "session_id": _orchestrator_state.get("session_id", ""),
        "has_pending_input": _orchestrator_state.get("pending_human_input") is not None,
        "iteration": _orchestrator_state.get("iteration", 0),
        "research_complete": _orchestrator_state.get("research_complete", False),
    })


async def send_orchestrator_message(request: Request):
    """
    Send a message to the orchestrator at any time.

    UNIFIED CONVERSATION: Primary interaction endpoint.
    Users can send messages without waiting for approval or phase transitions.

    Body:
        {
            "message": "user's message content"
        }

    Returns:
        {
            "success": true,
            "message_received": true,
            "session_id": "...",
            "phase": "current phase",
            "response": {
                "role": "system",
                "content": "orchestrator's response",
                "timestamp": "..."
            }
        }
    """
    try:
        body = await request.json()
        message = body.get("message", "").strip()

        if not message:
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )

        # Get current orchestrator state
        global _orchestrator_state

        if not _orchestrator_state:
            return JSONResponse(
                status_code=404,
                content={"error": "No active orchestrator session"}
            )

        # Process the message using the enhanced function
        from agents.orchestrator import process_user_message

        updates = process_user_message(_orchestrator_state, message)

        # Apply updates to orchestrator state
        _orchestrator_state.update(updates)

        # Save updated state
        _save_session_state()

        logger.info(f"Processed user message in session {_orchestrator_state.get('session_id')}")

        # Extract response message (if any) from updates
        response_messages = updates.get("messages", [])
        system_response = None
        if len(response_messages) > 1:
            # Second message is the system response
            system_response = response_messages[1]

        return JSONResponse(content={
            "success": True,
            "message_received": True,
            "session_id": _orchestrator_state.get("session_id"),
            "phase": _orchestrator_state.get("phase"),
            "response": system_response,
            "message": "Message processed successfully"
        })

    except Exception as e:
        logger.error(f"Error processing user message: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process message: {str(e)}"}
        )


async def get_conversation_history(request: Request):
    """
    Get full conversation history for the current session.

    UNIFIED CONVERSATION: Returns all messages exchanged with the orchestrator.

    Returns:
        {
            "session_id": "...",
            "messages": [
                {
                    "role": "user" | "system",
                    "content": "...",
                    "timestamp": "...",
                    "phase": "...",
                    "type": "message" | "question" | "answer" | "approval"
                }
            ],
            "spec_provided": true/false,
            "phase": "current_phase",
            "has_pending_input": true/false
        }
    """
    try:
        # Refresh state from file
        _refresh_session_state()

        global _orchestrator_state

        if not _orchestrator_state:
            return JSONResponse(
                status_code=404,
                content={"error": "No active session"}
            )

        # Get conversation messages from state
        messages = _orchestrator_state.get("messages", [])

        # Check if spec was provided
        spec_provided = bool(_orchestrator_state.get("spec_content"))

        return JSONResponse(content={
            "session_id": _orchestrator_state.get("session_id", ""),
            "messages": messages,
            "spec_provided": spec_provided,
            "phase": _orchestrator_state.get("phase", "INIT"),
            "has_pending_input": _orchestrator_state.get("pending_human_input") is not None,
            "pending_human_input": _orchestrator_state.get("pending_human_input"),
        })

    except Exception as e:
        logger.error(f"Error fetching conversation history: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to fetch conversation history: {str(e)}"}
        )


# =============================================================================
# DIALECTIC WEBSOCKET (Phase 3 - Real-time Updates)
# =============================================================================

# WebSocket connections for dialectic updates (session_id -> websocket)
_dialectic_ws_connections: Dict[str, WebSocket] = {}

# WebSocket connections for audio streaming (session_id -> websocket)
_audio_ws_connections: Dict[str, WebSocket] = {}


async def dialectic_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time dialectic phase updates.

    Protocol:
    1. Client connects with ?session_id=<id>
    2. Server sends current dialectic state (phase, questions, etc.)
    3. Server streams updates as orchestrator progresses
    4. Client can send answer submissions
    """
    session_id = websocket.query_params.get("session_id", "")
    await websocket.accept()

    if session_id:
        _dialectic_ws_connections[session_id] = websocket

    try:
        # Send initial state
        global _orchestrator_state
        _ensure_session_state_loaded()

        # Get live phase from diagnostics if available
        live_phase = _orchestrator_state.get("phase", "INIT")
        if session_id:
            diagnostics_info = _get_current_phase_from_diagnostics(session_id)
            if diagnostics_info and diagnostics_info.get("phase"):
                live_phase = diagnostics_info["phase"]

        initial_state = {
            "type": "state_update",
            "data": {
                "session_id": session_id or _orchestrator_state.get("session_id", ""),
                "status": "active",
                "current_phase": live_phase,
                "ambiguities": [],
                "questions": _format_questions_for_ws(_orchestrator_state.get("clarification_questions", [])),
                "answers": [],
                "resolved_count": 0,
                "total_count": len(_orchestrator_state.get("clarification_questions", [])),
                "started_at": _orchestrator_state.get("created_at", ""),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
        await websocket.send_json(initial_state)

        # Listen for client messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=30.0
                )

                msg_type = data.get("type", "")

                if msg_type == "submit_answer":
                    # Process answer submission
                    answers = data.get("data", {}).get("answers", [])
                    answer_text = "\n".join([
                        f"{i+1}. {ans.get('answer', '')}"
                        for i, ans in enumerate(answers)
                    ])

                    _orchestrator_state["human_response"] = answer_text
                    _orchestrator_state["pending_human_input"] = None

                    # Try to resume orchestrator
                    orchestrator = get_orchestrator()
                    if orchestrator and session_id:
                        try:
                            result = orchestrator.resume(session_id, human_response=answer_text)
                            _orchestrator_state.update(result)
                        except Exception as e:
                            _orchestrator_state["phase"] = "research"

                    # Send updated state
                    updated_phase = _orchestrator_state.get("phase", "research")
                    await websocket.send_json({
                        "type": "state_update",
                        "data": {
                            "session_id": session_id,
                            "status": "active",
                            "current_phase": updated_phase,
                            "ambiguities": [],
                            "questions": [],
                            "answers": [],
                            "resolved_count": len(answers),
                            "total_count": len(answers),
                            "started_at": _orchestrator_state.get("created_at", ""),
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }
                    })

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        pass
    finally:
        if session_id in _dialectic_ws_connections:
            del _dialectic_ws_connections[session_id]


def _format_questions_for_ws(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format clarification questions for WebSocket transmission.

    Matches TypeScript ClarificationQuestion interface:
    {
        id: string;
        question: string;
        context?: string;
        options?: string[];
        requires_freeform?: boolean;
        related_ambiguity_ids?: string[];
        priority: 'low' | 'medium' | 'high';
    }
    """
    formatted = []
    for i, q in enumerate(questions):
        question_text = q.get("question", q.get("phrase", ""))
        suggested = q.get("suggested_answer")

        formatted.append({
            "id": f"q_{i}",
            "question": question_text,
            "context": q.get("category", ""),
            "options": [suggested] if suggested else [],
            "requires_freeform": True,
            "priority": "medium",
        })
    return formatted


async def broadcast_dialectic_update(session_id: str, update: Dict[str, Any]) -> None:
    """
    Broadcast dialectic state update to connected client.

    Args:
        session_id: Session identifier
        update: Dict containing DialecticState fields:
            - session_id: str
            - status: 'active' | 'paused' | 'completed' | 'abandoned'
            - current_phase: str
            - ambiguities: List[AmbiguityMarker]
            - questions: List[ClarificationQuestion]
            - answers: List[ClarificationAnswer]
            - resolved_count: int
            - total_count: int
            - started_at: ISO timestamp
            - updated_at: ISO timestamp
    """
    if session_id in _dialectic_ws_connections:
        ws = _dialectic_ws_connections[session_id]
        try:
            await ws.send_json({
                "type": "state_update",
                "data": update
            })
        except Exception:
            del _dialectic_ws_connections[session_id]


async def broadcast_phase_change(session_id: str, phase: str, progress_info: Optional[Dict[str, Any]] = None) -> None:
    """
    Broadcast phase change to connected WebSocket client.

    Called by orchestrator/diagnostics when phase changes.

    Args:
        session_id: Session identifier
        phase: New phase name (e.g., 'research', 'plan', 'build')
        progress_info: Optional dict with progress details:
            - last_llm_call: str - Name of last LLM schema called
            - llm_call_count: int - Total LLM calls in session
            - iteration: int - Current iteration number
    """
    global _orchestrator_state

    # Update global state
    _orchestrator_state["phase"] = phase
    if progress_info:
        _orchestrator_state.update(progress_info)

    # Prepare state update message
    update = {
        "session_id": session_id,
        "status": "active",
        "current_phase": phase,
        "ambiguities": [],
        "questions": _format_questions_for_ws(_orchestrator_state.get("clarification_questions", [])),
        "answers": [],
        "resolved_count": 0,
        "total_count": len(_orchestrator_state.get("clarification_questions", [])),
        "started_at": _orchestrator_state.get("created_at", ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # Add progress info if available
    if progress_info:
        update["progress"] = progress_info

    await broadcast_dialectic_update(session_id, update)


# =============================================================================
# ORCHESTRATOR SESSION MANAGEMENT
# =============================================================================

import uuid as _uuid


async def start_orchestrator_session(request: Request) -> JSONResponse:
    """
    Start a new TDD orchestrator session.

    Body:
        {
            "requirement_text": "Create a function that...",
            "requirements": ["Must be fast", "Must handle errors"]  # optional
        }

    Response:
        {
            "session_id": "...",
            "status": "started",
            "phase": "DIALECTIC"
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    requirement_text = body.get("requirement_text", "").strip()
    requirements = body.get("requirements", [])

    # Validate requirement_text
    if not requirement_text:
        return error_response("requirement_text is required and cannot be empty", status_code=400)

    if len(requirement_text) > 10000:
        return error_response("requirement_text too long (max 10000 characters)", status_code=400)

    # Validate requirements array
    if requirements and not isinstance(requirements, list):
        return error_response("requirements must be an array", status_code=400)

    # Validate each requirement
    sanitized_requirements = []
    for i, req in enumerate(requirements):
        if not isinstance(req, str):
            return error_response(f"Requirement {i} must be a string", status_code=400)

        req_text = req.strip()
        if not req_text:
            continue  # Skip empty requirements

        if len(req_text) > 1000:
            return error_response(f"Requirement {i} too long (max 1000 characters)", status_code=400)

        sanitized_requirements.append(req_text)

    requirements = sanitized_requirements

    # Generate session ID
    session_id = f"session_{_uuid.uuid4().hex[:12]}"
    task_id = f"task_{_uuid.uuid4().hex[:8]}"

    # Update global state for polling endpoints
    global _orchestrator_state
    _orchestrator_state = {
        "session_id": session_id,
        "task_id": task_id,
        "phase": "DIALECTIC",
        "spec": requirement_text,
        "requirements": requirements,
        "clarification_questions": [],
        "pending_human_input": None,
    }

    # Get orchestrator and start in background (non-blocking)
    orchestrator = get_orchestrator()
    if orchestrator:
        # For now, return immediately and let client poll or use WebSocket
        # Full async background execution would require more infrastructure
        _orchestrator_state["orchestrator_available"] = True
    else:
        _orchestrator_state["orchestrator_available"] = False

    return JSONResponse({
        "session_id": session_id,
        "task_id": task_id,
        "status": "started",
        "phase": "DIALECTIC",
        "orchestrator_available": _orchestrator_state.get("orchestrator_available", False),
    }, status_code=201)


async def get_current_session(request: Request) -> JSONResponse:
    """
    Get the current session state with live progress from diagnostics.

    This endpoint returns the session state, including any spec loaded at startup.
    It also queries the diagnostics log to get the current phase (which may differ
    from the initial phase if the orchestrator has progressed).

    Response:
        {
            "session_id": "...",
            "has_spec": true,
            "title": "Project Title",
            "description": "...",
            "spec_content": "...",
            "phase": "BUILD",  // Live phase from diagnostics (UPPERCASE)
            "initial_phase": "PLAN",  // Phase at session start (UPPERCASE)
            "requirements": ["req1", "req2"],
            "progress": {
                "last_llm_call": "CodeGeneration",
                "llm_call_count": 5
            }
        }
    """
    # Refresh from file to pick up orchestrator updates (questions, phase changes)
    _refresh_session_state()
    _ensure_session_state_loaded()

    if not _orchestrator_state or not _orchestrator_state.get("session_id"):
        return JSONResponse({
            "session_id": None,
            "has_spec": False,
            "title": "",
            "description": "",
            "spec_content": "",
            "phase": "init",
            "initial_phase": "init",
            "requirements": [],
            "progress": None,
            "message": "No active session. Start fresh or load a spec.",
        })

    session_id = _orchestrator_state.get("session_id")
    initial_phase = _orchestrator_state.get("phase", "INIT")

    # Get live phase from diagnostics
    diagnostics_info = _get_current_phase_from_diagnostics(session_id)

    # Use live phase if available, otherwise fall back to initial
    live_phase = initial_phase
    progress = None
    if diagnostics_info:
        if diagnostics_info.get("phase"):
            live_phase = diagnostics_info["phase"]
        progress = {
            "last_llm_call": diagnostics_info.get("last_llm_call"),
            "llm_call_count": diagnostics_info.get("llm_call_count", 0),
        }

    # Get clarification questions if in dialectic/clarification phase (UPPERCASE)
    questions = []
    if live_phase in ("DIALECTIC", "CLARIFICATION"):
        raw_questions = _orchestrator_state.get("clarification_questions", [])
        for i, q in enumerate(raw_questions):
            # Format questions for frontend - ensure they match ClarificationQuestion interface
            formatted_q = {
                "id": f"q_{i}",
                "question": q.get("question", q.get("phrase", "")),
                "context": q.get("category", ""),
                "options": [],
                "requires_freeform": True,
                "priority": "medium",  # Default priority
            }
            # Add suggested answer as an option if available
            if q.get("suggested_answer"):
                formatted_q["options"] = [q.get("suggested_answer")]
            questions.append(formatted_q)

    return JSONResponse({
        "session_id": session_id,
        "has_spec": bool(_orchestrator_state.get("spec_content")),
        "title": _orchestrator_state.get("title", ""),
        "description": _orchestrator_state.get("description", ""),
        "spec_content": _orchestrator_state.get("spec_content", ""),
        "phase": live_phase,
        "initial_phase": initial_phase,
        "requirements": _orchestrator_state.get("requirements", []),
        "created_at": _orchestrator_state.get("created_at", ""),
        "progress": progress,
        "questions": questions,
        "pending_human_input": _orchestrator_state.get("pending_human_input"),
    })


async def run_orchestrator_phase(request: Request) -> JSONResponse:
    """
    Run the next phase of an orchestrator session.

    Body:
        {
            "session_id": "...",
            "phase": "DIALECTIC"  # optional - run specific phase (UPPERCASE)
        }

    This allows step-by-step execution for debugging/testing.
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    session_id = body.get("session_id")
    requested_phase = body.get("phase")

    if not session_id:
        return error_response("session_id is required", status_code=400)

    global _orchestrator_state
    if _orchestrator_state.get("session_id") != session_id:
        return error_response(f"Session {session_id} not found", status_code=404)

    orchestrator = get_orchestrator()
    if not orchestrator:
        return error_response("Orchestrator not available", status_code=503)

    try:
        # Run orchestrator with current state
        result = orchestrator.run(
            session_id=session_id,
            task_id=_orchestrator_state.get("task_id", ""),
            spec=_orchestrator_state.get("spec", ""),
            requirements=_orchestrator_state.get("requirements", []),
            max_iterations=3,
            fresh=False,  # Resume existing session
        )

        # Update global state
        _orchestrator_state.update(result)

        return JSONResponse({
            "session_id": session_id,
            "phase": result.get("phase", "unknown"),
            "has_questions": len(result.get("clarification_questions", [])) > 0,
            "questions": _format_questions_for_ws(result.get("clarification_questions", [])),
            "final_status": result.get("final_status"),
        })

    except Exception as e:
        return error_response(f"Orchestrator error: {e}", status_code=500)


# =============================================================================
# GRAPH VALIDATION ENDPOINT
# =============================================================================

async def validate_graph_endpoint(request: Request) -> JSONResponse:
    """
    Validate the current graph state for integrity and health.

    This endpoint runs comprehensive graph validation checks using the
    core.graph_invariants module, including:
    - Handshaking Lemma (sum(in_degree) == sum(out_degree) == |E|)
    - DAG Acyclicity (no cycles)
    - Balis Degree (all sources reach all sinks)
    - Stratification (topological type ordering)
    - Cyclomatic complexity
    - Articulation points (critical nodes)

    Query params:
        detailed: bool (default=false) - Include verbose metrics and node details
        fix_issues: bool (default=false) - Attempt auto-repair of fixable issues
        raise_on_error: bool (default=false) - Raise exception on ERROR severity

    Response:
        {
            "is_valid": true,
            "timestamp": "2025-12-08T...",
            "node_count": 42,
            "edge_count": 58,
            "errors": [
                {
                    "invariant": "handshaking_lemma",
                    "severity": "error",
                    "message": "sum(in_degree)=50 != sum(out_degree)=48",
                    "nodes_involved": ["node_id_1"],
                    "edges_involved": []
                }
            ],
            "warnings": [
                {
                    "invariant": "balis_degree",
                    "severity": "warning",
                    "message": "3 source-sink pair(s) unreachable",
                    "nodes_involved": [],
                    "edges_involved": [["REQ-1", "CODE-5"]]
                }
            ],
            "metrics": {
                "node_count": 42,
                "edge_count": 58,
                "cyclomatic_complexity": 5,
                "weakly_connected_components": 1,
                "articulation_point_count": 3,
                "unreachable_pairs": 3
            },
            "details": {
                "articulation_points": ["node_1", "node_3"],
                "bridge_edges": [["node_1", "node_2"]],
                "root_nodes": ["REQ-1", "REQ-2"],
                "leaf_nodes": ["TEST-1", "TEST-2"]
            }
        }

    Status codes:
        200: Validation completed (check is_valid field)
        400: Invalid query parameters
        500: Database unavailable or validation error
        503: Database not initialized
    """
    try:
        from core.graph_invariants import validate_graph, GraphInvariants
        import logging
        logger = logging.getLogger("paragon.api.validate")

        # Parse query parameters
        detailed = request.query_params.get("detailed", "false").lower() == "true"
        fix_issues = request.query_params.get("fix_issues", "false").lower() == "true"
        raise_on_error = request.query_params.get("raise_on_error", "false").lower() == "true"

        # Get database instance
        db = get_db()
        if db is None:
            logger.error("Database not initialized")
            return error_response("Database not initialized", status_code=503)

        # Log validation request
        logger.info(
            f"Graph validation requested: detailed={detailed}, "
            f"fix_issues={fix_issues}, raise_on_error={raise_on_error}"
        )

        # Check if graph is empty
        if db.is_empty:
            logger.info("Graph is empty, returning valid state")
            return JSONResponse({
                "is_valid": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node_count": 0,
                "edge_count": 0,
                "errors": [],
                "warnings": [],
                "metrics": {
                    "node_count": 0,
                    "edge_count": 0,
                    "cyclomatic_complexity": 0,
                    "weakly_connected_components": 0,
                },
                "message": "Graph is empty (no nodes or edges)"
            })

        # Access the internal rustworkx graph
        # The validate_graph function expects the rx.PyDiGraph directly
        graph = db._graph
        inv_map = db._inv_map  # For node ID mapping in error messages

        # Run validation
        try:
            report = validate_graph(
                graph,
                inv_map=inv_map,
                type_order=None,  # Use default Paragon ordering
                raise_on_error=raise_on_error
            )
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return error_response(
                f"Validation error: {str(e)}",
                status_code=500
            )

        # Log validation results
        error_count = len(report.errors)
        warning_count = len(report.warnings)
        logger.info(
            f"Validation completed: valid={report.valid}, "
            f"errors={error_count}, warnings={warning_count}"
        )

        # Build response
        response_data = {
            "is_valid": report.valid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_count": db.node_count,
            "edge_count": db.edge_count,
            "errors": [
                {
                    "invariant": v.invariant,
                    "severity": v.severity.value,
                    "message": v.message,
                    "nodes_involved": v.nodes_involved if v.nodes_involved else [],
                    "edges_involved": v.edges_involved if v.edges_involved else [],
                }
                for v in report.errors
            ],
            "warnings": [
                {
                    "invariant": v.invariant,
                    "severity": v.severity.value,
                    "message": v.message,
                    "nodes_involved": v.nodes_involved if v.nodes_involved else [],
                    "edges_involved": v.edges_involved if v.edges_involved else [],
                }
                for v in report.warnings
            ],
            "metrics": report.metrics,
        }

        # Add detailed information if requested
        if detailed:
            details = {}

            # Get articulation points (critical nodes)
            if graph.num_nodes() < 1000:
                try:
                    articulation_indices = GraphInvariants.get_articulation_points(graph)
                    articulation_ids = [
                        inv_map.get(idx, f"idx_{idx}")
                        for idx in articulation_indices
                    ]
                    details["articulation_points"] = articulation_ids
                except Exception as e:
                    logger.warning(f"Could not compute articulation points: {e}")
                    details["articulation_points"] = []

            # Get bridge edges (critical dependencies)
            if graph.num_edges() < 1000:
                try:
                    bridge_edges = GraphInvariants.get_bridge_edges(graph, inv_map)
                    details["bridge_edges"] = list(bridge_edges)
                except Exception as e:
                    logger.warning(f"Could not compute bridge edges: {e}")
                    details["bridge_edges"] = []

            # Get root and leaf nodes
            try:
                root_nodes = db.get_root_nodes()
                leaf_nodes = db.get_leaf_nodes()
                details["root_nodes"] = [n.id for n in root_nodes]
                details["leaf_nodes"] = [n.id for n in leaf_nodes]
            except Exception as e:
                logger.warning(f"Could not get root/leaf nodes: {e}")
                details["root_nodes"] = []
                details["leaf_nodes"] = []

            # Add node type distribution
            try:
                type_counts = {}
                for node in db.iter_nodes():
                    node_type = node.type
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                details["node_type_distribution"] = type_counts
            except Exception as e:
                logger.warning(f"Could not compute type distribution: {e}")
                details["node_type_distribution"] = {}

            # Add edge type distribution
            try:
                edge_type_counts = {}
                for edge in db.get_all_edges():
                    edge_type = edge.type
                    edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
                details["edge_type_distribution"] = edge_type_counts
            except Exception as e:
                logger.warning(f"Could not compute edge type distribution: {e}")
                details["edge_type_distribution"] = {}

            response_data["details"] = details

        # Handle fix_issues parameter (auto-repair)
        if fix_issues and not report.valid:
            # Note: Auto-repair is not implemented in this version
            # This is a placeholder for future enhancement
            response_data["fix_attempted"] = False
            response_data["fix_message"] = (
                "Auto-repair is not yet implemented. "
                "Manual intervention required for ERROR-level violations."
            )
            logger.warning("Auto-repair requested but not implemented")

        return JSONResponse(response_data)

    except ImportError as e:
        logger.error(f"Graph invariants module not available: {e}")
        return error_response(
            "Graph validation module not available",
            status_code=503
        )
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return error_response(
            f"Validation failed: {str(e)}",
            status_code=500
        )


# =============================================================================
# EDGE CASE ENDPOINTS
# =============================================================================

_edge_case_collector = None


def _get_edge_case_collector():
    """Get or create the EdgeCaseCollector instance."""
    global _edge_case_collector
    if _edge_case_collector is None:
        try:
            from infrastructure.edge_cases import EdgeCaseCollector
            _edge_case_collector = EdgeCaseCollector()
        except ImportError:
            pass
    return _edge_case_collector


async def list_edge_cases(request: Request) -> JSONResponse:
    """
    List edge cases with optional filters.

    Query params:
        project_id: Filter by project
        session_id: Filter by session
        category: Filter by category (e.g., "parser_divergence")
        severity: Filter by severity ("low", "medium", "high", "critical")
        resolved: Filter by resolution status (true/false)
        limit: Max results (default 100)
    """
    collector = _get_edge_case_collector()
    if not collector:
        return error_response("Edge case collector not available", status_code=503)

    try:
        params = dict(request.query_params)

        # Parse query params
        project_id = params.get("project_id")
        session_id = params.get("session_id")
        category = params.get("category")
        severity = params.get("severity")
        resolved_str = params.get("resolved")
        limit = int(params.get("limit", "100"))

        resolved = None
        if resolved_str is not None:
            resolved = resolved_str.lower() == "true"

        # Query
        cases = collector.query(
            project_id=project_id,
            session_id=session_id,
            category=category,
            severity=severity,
            resolved=resolved,
            limit=limit,
        )

        return JSONResponse({
            "edge_cases": [
                {
                    "edge_case_id": c.edge_case_id,
                    "node_id": c.node_id,
                    "session_id": c.session_id,
                    "project_id": c.project_id,
                    "categories": c.categories,
                    "severity": c.severity,
                    "source": c.source,
                    "code_snippet": c.code_snippet[:200] + "..." if len(c.code_snippet) > 200 else c.code_snippet,
                    "description": c.description,
                    "detected_at": c.detected_at,
                    "resolved": c.resolved,
                    "flagged_by": c.flagged_by,
                    "flag_reason": c.flag_reason,
                }
                for c in cases
            ],
            "count": len(cases),
        })

    except Exception as e:
        return error_response(f"Query error: {e}", status_code=500)


async def get_edge_case(request: Request) -> JSONResponse:
    """Get a specific edge case by ID."""
    collector = _get_edge_case_collector()
    if not collector:
        return error_response("Edge case collector not available", status_code=503)

    edge_case_id = request.path_params.get("edge_case_id", "")

    try:
        case = collector.store.get(edge_case_id)
        if not case:
            return error_response(f"Edge case {edge_case_id} not found", status_code=404)

        return JSONResponse({
            "edge_case_id": case.edge_case_id,
            "node_id": case.node_id,
            "session_id": case.session_id,
            "project_id": case.project_id,
            "categories": case.categories,
            "severity": case.severity,
            "source": case.source,
            "code_snippet": case.code_snippet,
            "description": case.description,
            "detection_details": case.detection_details,
            "detected_at": case.detected_at,
            "resolved": case.resolved,
            "resolution_notes": case.resolution_notes,
            "resolved_at": case.resolved_at,
            "flagged_by": case.flagged_by,
            "flag_reason": case.flag_reason,
        })

    except Exception as e:
        return error_response(f"Error: {e}", status_code=500)


async def flag_edge_case(request: Request) -> JSONResponse:
    """
    Manually flag something as an edge case.

    Body:
        node_id: str - Node ID (or "manual" for unattached)
        code_snippet: str - The code or content
        reason: str - Why this is interesting
        flagged_by: str - Who is flagging
        session_id: str (optional)
        project_id: str (optional)
    """
    collector = _get_edge_case_collector()
    if not collector:
        return error_response("Edge case collector not available", status_code=503)

    try:
        body = await request.json()

        node_id = body.get("node_id", "manual").strip() if isinstance(body.get("node_id"), str) else "manual"
        code_snippet = body.get("code_snippet", "").strip() if isinstance(body.get("code_snippet"), str) else ""
        reason = body.get("reason", "").strip() if isinstance(body.get("reason"), str) else ""
        flagged_by = body.get("flagged_by", "unknown").strip() if isinstance(body.get("flagged_by"), str) else "unknown"
        session_id = body.get("session_id", "").strip() if isinstance(body.get("session_id"), str) else ""
        project_id = body.get("project_id", "").strip() if isinstance(body.get("project_id"), str) else ""

        # Validate required fields
        if not reason:
            return error_response("reason is required and cannot be empty", status_code=400)

        if len(reason) > 5000:
            return error_response("reason too long (max 5000 characters)", status_code=400)

        if len(node_id) > 100:
            return error_response("node_id too long (max 100 characters)", status_code=400)

        if len(code_snippet) > 50000:
            return error_response("code_snippet too long (max 50000 characters)", status_code=400)

        if len(flagged_by) > 100:
            return error_response("flagged_by too long (max 100 characters)", status_code=400)

        if session_id and len(session_id) > 100:
            return error_response("session_id too long (max 100 characters)", status_code=400)

        if project_id and len(project_id) > 100:
            return error_response("project_id too long (max 100 characters)", status_code=400)

        case = collector.flag_manually(
            node_id=node_id,
            code_snippet=code_snippet,
            reason=reason,
            flagged_by=flagged_by,
            session_id=session_id,
            project_id=project_id,
        )

        return JSONResponse({
            "success": True,
            "edge_case_id": case.edge_case_id,
            "message": "Edge case flagged successfully",
        })

    except Exception as e:
        return error_response(f"Error: {e}", status_code=500)


async def resolve_edge_case(request: Request) -> JSONResponse:
    """
    Mark an edge case as resolved.

    Body:
        notes: str (optional) - Resolution notes
    """
    collector = _get_edge_case_collector()
    if not collector:
        return error_response("Edge case collector not available", status_code=503)

    edge_case_id = request.path_params.get("edge_case_id", "")

    try:
        body = await request.json()
        notes = body.get("notes", "")

        success = collector.store.mark_resolved(edge_case_id, notes)
        if not success:
            return error_response(f"Edge case {edge_case_id} not found", status_code=404)

        return JSONResponse({
            "success": True,
            "edge_case_id": edge_case_id,
            "message": "Edge case resolved",
        })

    except Exception as e:
        return error_response(f"Error: {e}", status_code=500)


async def edge_case_summary(request: Request) -> JSONResponse:
    """Get summary statistics of edge cases."""
    collector = _get_edge_case_collector()
    if not collector:
        return error_response("Edge case collector not available", status_code=503)

    try:
        summary = collector.get_summary()
        return JSONResponse(summary)

    except Exception as e:
        return error_response(f"Error: {e}", status_code=500)


# =============================================================================
# SPEECH-TO-TEXT ENDPOINTS
# =============================================================================

async def transcribe_audio_file(request: Request) -> JSONResponse:
    """
    Transcribe an uploaded audio file.

    Body:
        Multipart form data with 'audio' file
        Optional 'language' field (e.g., 'en', 'es')

    Response:
        {
            "text": "Full transcription...",
            "segments": [
                {
                    "text": "Segment text",
                    "start": 0.0,
                    "end": 2.5,
                    "confidence": 0.95
                }
            ],
            "language": "en",
            "duration": 10.5,
            "model": "faster-whisper-base"
        }
    """
    try:
        from infrastructure.speech_to_text import get_stt_service
        import tempfile

        stt = get_stt_service()
        if not stt.available:
            return error_response(
                "Speech-to-text service not available. Install with: pip install faster-whisper",
                status_code=503
            )

        # Parse multipart form data
        form = await request.form()
        audio_file = form.get("audio")
        language = form.get("language")

        if not audio_file:
            return error_response("No audio file provided", status_code=400)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio_file.read())
            tmp_path = Path(tmp.name)

        try:
            # Transcribe
            result = stt.transcribe_file(tmp_path, language=language)

            if not result:
                return error_response("Transcription failed", status_code=500)

            # Convert to dict for JSON response
            return JSONResponse({
                "text": result.text,
                "segments": [
                    {
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                        "confidence": seg.confidence,
                        "is_final": seg.is_final,
                        "language": seg.language,
                    }
                    for seg in result.segments
                ],
                "language": result.language,
                "duration": result.duration,
                "model": result.model,
            })

        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    except ImportError:
        return error_response(
            "Speech-to-text dependencies not installed. Install with: pip install faster-whisper numpy",
            status_code=503
        )
    except Exception as e:
        return error_response(f"Transcription error: {e}", status_code=500)


async def audio_streaming_websocket(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time audio streaming and transcription.

    Protocol:
    1. Client connects with ?session_id=<id> (optional)
    2. Client sends binary audio data (16-bit PCM, 16kHz recommended)
    3. Server streams back transcript segments as JSON
    4. Client sends {"type": "stop"} to end transcription

    Message format (server -> client):
        {
            "type": "transcript",
            "data": {
                "text": "Segment text",
                "start": 0.0,
                "end": 2.5,
                "confidence": 0.95,
                "is_final": false,
                "language": "en"
            }
        }

    Audio format:
    - Sample rate: 16000 Hz (recommended)
    - Channels: 1 (mono)
    - Sample width: 16-bit PCM
    - Encoding: Linear PCM

    Example client (JavaScript):
        const ws = new WebSocket('ws://localhost:8000/ws/audio');
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = (event) => {
            ws.send(event.data);
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            if (msg.type === 'transcript') {
                console.log('Transcript:', msg.data.text);
            }
        };

        mediaRecorder.start(1000);  // Send chunks every 1 second
    """
    session_id = websocket.query_params.get("session_id", "")
    await websocket.accept()

    if session_id:
        _audio_ws_connections[session_id] = websocket

    try:
        from infrastructure.speech_to_text import get_stt_service

        stt = get_stt_service()

        if not stt.available:
            await websocket.send_json({
                "type": "error",
                "message": "Speech-to-text service not available"
            })
            await websocket.close()
            return

        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "Audio streaming ready",
            "model": f"faster-whisper-{stt.model_size}"
        })

        # Create async generator for audio stream
        async def audio_stream():
            while True:
                try:
                    # Receive binary audio data or JSON commands
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=30.0
                    )

                    # Handle binary audio data
                    if "bytes" in message:
                        yield message["bytes"]

                    # Handle JSON commands (e.g., stop)
                    elif "text" in message:
                        data = msgspec.json.decode(message["text"])
                        if data.get("type") == "stop":
                            break

                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    await websocket.send_json({"type": "ping"})
                except WebSocketDisconnect:
                    break

        # Stream transcription
        async for segment in stt.transcribe_streaming(audio_stream()):
            try:
                await websocket.send_json({
                    "type": "transcript",
                    "data": {
                        "text": segment.text,
                        "start": segment.start,
                        "end": segment.end,
                        "confidence": segment.confidence,
                        "is_final": segment.is_final,
                        "language": segment.language,
                    }
                })
            except Exception as e:
                logger.error(f"Failed to send transcript segment: {e}")
                break

        # Send completion message
        await websocket.send_json({
            "type": "complete",
            "message": "Transcription complete"
        })

    except ImportError:
        await websocket.send_json({
            "type": "error",
            "message": "Speech-to-text dependencies not installed"
        })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Audio streaming error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Transcription error: {e}"
            })
        except Exception:
            pass
    finally:
        if session_id in _audio_ws_connections:
            del _audio_ws_connections[session_id]


async def stt_status(request: Request) -> JSONResponse:
    """
    Check speech-to-text service status.

    Response:
        {
            "available": true,
            "model": "faster-whisper-base",
            "device": "cpu",
            "vad_enabled": true,
            "llm_correction_enabled": false
        }
    """
    try:
        from infrastructure.speech_to_text import get_stt_service

        stt = get_stt_service()

        return JSONResponse({
            "available": stt.available,
            "model": f"faster-whisper-{stt.model_size}" if stt.available else None,
            "device": stt.device,
            "vad_enabled": stt.enable_vad,
            "llm_correction_enabled": stt.enable_llm_correction,
        })

    except ImportError:
        return JSONResponse({
            "available": False,
            "error": "faster-whisper not installed"
        })
    except Exception as e:
        return error_response(f"Status check failed: {e}", status_code=500)


# =============================================================================
# PARALLEL PIPELINE DETECTION (Layer 8 Physics)
# =============================================================================

# Global whitelist for parallel pipeline detection
_parallel_pipeline_whitelist: Set[str] = set()
_whitelist_file = Path(__file__).parent.parent / "workspace" / "parallel_pipeline_whitelist.json"


def _load_whitelist():
    """Load whitelist from file."""
    global _parallel_pipeline_whitelist
    if _whitelist_file.exists():
        try:
            import json
            with open(_whitelist_file, "r") as f:
                _parallel_pipeline_whitelist = set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load whitelist: {e}")


def _save_whitelist():
    """Save whitelist to file."""
    try:
        import json
        _whitelist_file.parent.mkdir(exist_ok=True)
        with open(_whitelist_file, "w") as f:
            json.dump(list(_parallel_pipeline_whitelist), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save whitelist: {e}")


# Load whitelist on startup
_load_whitelist()


async def get_parallel_pipeline_detections(request: Request) -> JSONResponse:
    """
    GET /api/physics/parallel-pipelines

    Detect parallel pipelines in the codebase.

    Query params:
        force_rebuild: If true, ignore cache and rebuild from scratch
    """
    try:
        from core.parallel_pipeline_detector import ParallelPipelineDetector

        force_rebuild = request.query_params.get("force_rebuild", "false").lower() == "true"
        repo_path = Path(__file__).parent.parent

        detector = ParallelPipelineDetector(repo_path)
        candidates = detector.detect(force_rebuild=force_rebuild)

        # Filter out whitelisted files
        candidates = [
            c for c in candidates
            if c.file_path not in _parallel_pipeline_whitelist
        ]

        return JSONResponse({
            "count": len(candidates),
            "candidates": [msgspec.to_builtins(c) for c in candidates],
            "whitelist_count": len(_parallel_pipeline_whitelist),
        })

    except Exception as e:
        logger.error(f"Error detecting parallel pipelines: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def review_parallel_pipeline_detection(request: Request) -> JSONResponse:
    """
    POST /api/physics/parallel-pipelines/review

    Human feedback on a parallel pipeline detection.

    Request body:
        {
            "file_path": "path/to/file.py",
            "action": "DELETE" | "WHITELIST" | "MERGE" | "SKIP"
        }
    """
    try:
        body = await request.body()
        data = msgspec.json.decode(body)

        file_path = data.get("file_path")
        action = data.get("action")

        if not file_path or not action:
            return JSONResponse(
                {"error": "Missing file_path or action"},
                status_code=400
            )

        if action == "DELETE":
            # Queue for deletion (don't auto-delete)
            logger.info(f"Queued for deletion: {file_path}")
            # In production, this would create a GitHub issue or notification
            return JSONResponse({
                "status": "acknowledged",
                "message": f"Queued {file_path} for deletion review"
            })

        elif action == "WHITELIST":
            # Add to whitelist
            _parallel_pipeline_whitelist.add(file_path)
            _save_whitelist()
            logger.info(f"Whitelisted: {file_path}")
            return JSONResponse({
                "status": "acknowledged",
                "message": f"Added {file_path} to whitelist"
            })

        elif action == "MERGE":
            # Create merge issue
            logger.info(f"Create merge issue for: {file_path}")
            # In production, this would create a GitHub issue
            return JSONResponse({
                "status": "acknowledged",
                "message": f"Created merge issue for {file_path}"
            })

        elif action == "SKIP":
            return JSONResponse({
                "status": "acknowledged",
                "message": f"Skipped {file_path}"
            })

        else:
            return JSONResponse(
                {"error": f"Invalid action: {action}"},
                status_code=400
            )

    except Exception as e:
        logger.error(f"Error reviewing parallel pipeline: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_parallel_pipeline_whitelist(request: Request) -> JSONResponse:
    """
    GET /api/physics/parallel-pipelines/whitelist

    Get the current whitelist of files.
    """
    return JSONResponse({
        "whitelist": list(_parallel_pipeline_whitelist),
        "count": len(_parallel_pipeline_whitelist),
    })


# =============================================================================
# NOTIFICATION ENDPOINTS (Cross-Tab Notifications)
# =============================================================================

async def get_pending_notifications(request: Request) -> JSONResponse:
    """
    GET /api/notifications/pending

    Get all pending notifications for the current user/session.

    Response:
        {
            "notifications": [
                {
                    "notification_id": "...",
                    "type": "spec_updated",
                    "source": "orchestrator",
                    "message": "New requirement added",
                    "target_tab": "build",
                    "urgency": "info",
                    "metadata": {...},
                    "created_at": "...",
                    "read": false
                }
            ],
            "unread_count": 5
        }
    """
    db = get_db()

    # Query all NOTIFICATION nodes
    all_nodes = db.get_all_nodes()
    notification_nodes = [n for n in all_nodes if n.type == NodeType.NOTIFICATION.value]

    notifications = []
    unread_count = 0

    for node in notification_nodes:
        notification_data = node.data or {}
        read_by = notification_data.get("read_by", [])
        is_read = node.status == NodeStatus.VERIFIED.value

        if not is_read:
            unread_count += 1

        # Extract target tabs (may be list or single string)
        target_tabs = notification_data.get("target_tabs", [])
        if isinstance(target_tabs, str):
            target_tabs = [target_tabs]

        notifications.append({
            "notification_id": node.id,
            "type": notification_data.get("notification_type", "info"),
            "source": notification_data.get("source_component", "system"),
            "message": node.content,
            "target_tabs": target_tabs,
            "urgency": notification_data.get("urgency", "info"),
            "metadata": {
                "related_node_id": notification_data.get("related_node_id"),
                "action_required": notification_data.get("action_required", False),
            },
            "created_at": node.created_at,
            "read": is_read,
        })

    # Sort by created_at descending
    notifications.sort(key=lambda x: x["created_at"], reverse=True)

    return JSONResponse({
        "notifications": notifications,
        "unread_count": unread_count,
    })


async def mark_notification_read(request: Request) -> JSONResponse:
    """
    POST /api/notifications/{notification_id}/mark-read

    Mark a notification as read.

    Response:
        {
            "status": "success",
            "notification_id": "..."
        }
    """
    notification_id = request.path_params["notification_id"]
    db = get_db()

    # Get the notification node
    node = db.get_node(notification_id)
    if node is None:
        return error_response(f"Notification not found: {notification_id}", status_code=404)

    if node.type != NodeType.NOTIFICATION.value:
        return error_response(f"Node is not a notification: {notification_id}", status_code=400)

    # Mark as read by changing status to VERIFIED
    from agents.tools import update_node_status
    result = update_node_status(notification_id, NodeStatus.VERIFIED.value)

    if not result.get("success"):
        return error_response(f"Failed to mark notification as read: {result.get('error')}", status_code=500)

    return JSONResponse({
        "status": "success",
        "notification_id": notification_id,
    })


async def create_notification_endpoint(request: Request) -> JSONResponse:
    """
    POST /api/notifications/create

    Programmatic endpoint for creating notifications from orchestrator/agents.

    Body:
        {
            "notification_type": "spec_updated" | "research_complete" | "approval_needed" | "phase_changed",
            "message": "Notification message text",
            "target_tabs": ["build", "research", "specification"],
            "urgency": "info" | "warning" | "critical",
            "source_component": "orchestrator",
            "related_node_id": "optional-node-id",
            "action_required": false
        }

    Response:
        {
            "status": "success",
            "notification_id": "..."
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    notification_type = body.get("notification_type", "info")
    message = body.get("message", "").strip()
    target_tabs = body.get("target_tabs", [])
    urgency = body.get("urgency", "info")
    source_component = body.get("source_component", "system")
    related_node_id = body.get("related_node_id")
    action_required = body.get("action_required", False)

    if not message:
        return error_response("Message is required", status_code=400)

    if not target_tabs:
        return error_response("target_tabs is required", status_code=400)

    # Validate urgency
    if urgency not in ["info", "warning", "critical"]:
        return error_response("urgency must be one of: info, warning, critical", status_code=400)

    # Create notification node
    db = get_db()
    notification_node = NodeData.create(
        type=NodeType.NOTIFICATION.value,
        content=message,
        data={
            "notification_type": notification_type,
            "source_component": source_component,
            "target_tabs": target_tabs,
            "urgency": urgency,
            "related_node_id": related_node_id,
            "action_required": action_required,
            "read_by": [],
        },
        status=NodeStatus.PENDING.value,
        created_by="api",
    )

    db.add_node(notification_node)

    # Link to related node if provided
    if related_node_id:
        try:
            edge_data = EdgeData.create(
                source_id=notification_node.id,
                target_id=related_node_id,
                type=EdgeType.TRACES_TO.value,
            )
            db.add_edge(edge_data)
        except Exception as e:
            logger.warning(f"Failed to link notification to node {related_node_id}: {e}")

    # Publish notification event for WebSocket broadcasting
    if EVENT_BUS_AVAILABLE:
        from infrastructure.event_bus import publish_notification
        publish_notification(
            notification_type=notification_type,
            message=message,
            target_tabs=target_tabs,
            urgency=urgency,
            related_node_id=related_node_id,
            metadata={
                "notification_id": notification_node.id,
                "action_required": action_required,
            },
            source=source_component,
        )

    return JSONResponse({
        "status": "success",
        "notification_id": notification_node.id,
    }, status_code=201)


# =============================================================================
# DIALOGUE-TO-GRAPH HIGHLIGHTING (Interactive Visualization)
# =============================================================================

async def graph_highlight(request: Request) -> JSONResponse:
    """
    POST /api/graph/highlight

    Get nodes and edges to highlight based on a source node/message/edge.

    Request body:
    {
        "highlight_type": "message" | "node" | "edge",
        "source_id": str,
        "highlight_mode": "exact" | "related" | "dependent"
    }

    Response:
    {
        "nodes_to_highlight": List[str],
        "edges_to_highlight": List[{"source": str, "target": str}],
        "context": str
    }
    """
    try:
        body = await request.json()
        highlight_type = body.get("highlight_type", "node")
        source_id = body.get("source_id")
        highlight_mode = body.get("highlight_mode", "exact")

        if not source_id:
            return error_response("Missing source_id", 400)

        if highlight_type not in ("message", "node", "edge"):
            return error_response(f"Invalid highlight_type: {highlight_type}", 400)

        if highlight_mode not in ("exact", "related", "dependent"):
            return error_response(f"Invalid highlight_mode: {highlight_mode}", 400)

        db = get_db()

        # Determine nodes to highlight based on type
        nodes_to_highlight = []
        edges_to_highlight = []
        context = ""

        if highlight_type == "message":
            # Source is a MESSAGE or THREAD node - get all nodes it references
            if not db.has_node(source_id):
                return error_response(f"Message node not found: {source_id}", 404)

            # Use new method from message-to-node mapping
            referenced_nodes = db.get_nodes_from_message(source_id)
            referenced_node_ids = [n.id for n in referenced_nodes]

            # Apply highlighting mode to each referenced node
            all_nodes = set()
            for node_id in referenced_node_ids:
                try:
                    related = db.get_related_nodes(node_id, mode=highlight_mode)
                    all_nodes.update(related)
                except Exception:
                    # Node might not exist, skip it
                    continue

            nodes_to_highlight = sorted(all_nodes)
            context = f"Message {source_id[:8]} references {len(referenced_node_ids)} node(s)"

            # Get edges between highlighted nodes
            for i, src in enumerate(nodes_to_highlight):
                for tgt in nodes_to_highlight[i+1:]:
                    if db.has_edge(src, tgt):
                        edges_to_highlight.append({"source": src, "target": tgt})
                    elif db.has_edge(tgt, src):
                        edges_to_highlight.append({"source": tgt, "target": src})

        elif highlight_type == "node":
            # Source is a regular node
            if not db.has_node(source_id):
                return error_response(f"Node not found: {source_id}", 404)

            nodes_to_highlight = db.get_related_nodes(source_id, mode=highlight_mode)
            node = db.get_node(source_id)
            context = f"Node {source_id[:8]} ({node.type}) with {len(nodes_to_highlight)} related node(s)"

            # Get edges between highlighted nodes
            for i, src in enumerate(nodes_to_highlight):
                for tgt in nodes_to_highlight[i+1:]:
                    if db.has_edge(src, tgt):
                        edges_to_highlight.append({"source": src, "target": tgt})
                    elif db.has_edge(tgt, src):
                        edges_to_highlight.append({"source": tgt, "target": src})

        elif highlight_type == "edge":
            # Source is an edge identifier (source:target format)
            if ":" not in source_id:
                return error_response("Edge source_id must be in format 'source:target'", 400)

            edge_source, edge_target = source_id.split(":", 1)

            if not db.has_edge(edge_source, edge_target):
                return error_response(f"Edge not found: {edge_source} -> {edge_target}", 404)

            # Highlight both endpoints and their related nodes
            source_related = db.get_related_nodes(edge_source, mode=highlight_mode)
            target_related = db.get_related_nodes(edge_target, mode=highlight_mode)
            all_nodes = set(source_related + target_related)
            nodes_to_highlight = sorted(all_nodes)

            edges_to_highlight.append({"source": edge_source, "target": edge_target})
            context = f"Edge {edge_source[:8]} -> {edge_target[:8]} with {len(nodes_to_highlight)} related node(s)"

            # Get edges between highlighted nodes
            for i, src in enumerate(nodes_to_highlight):
                for tgt in nodes_to_highlight[i+1:]:
                    if db.has_edge(src, tgt):
                        edge_dict = {"source": src, "target": tgt}
                        if edge_dict not in edges_to_highlight:
                            edges_to_highlight.append(edge_dict)
                    elif db.has_edge(tgt, src):
                        edge_dict = {"source": tgt, "target": src}
                        if edge_dict not in edges_to_highlight:
                            edges_to_highlight.append(edge_dict)

        return JSONResponse({
            "nodes_to_highlight": nodes_to_highlight,
            "edges_to_highlight": edges_to_highlight,
            "context": context,
            "highlight_mode": highlight_mode,
            "highlight_type": highlight_type,
        })

    except ValueError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error in graph_highlight: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_node_reverse_connections(request: Request) -> JSONResponse:
    """
    GET /api/nodes/{node_id}/reverse-connections

    Get all MESSAGE nodes and edges that reference this node.

    Response:
    {
        "node_id": str,
        "referenced_in_messages": List[NodeData],
        "incoming_edges": List[Dict],
        "outgoing_edges": List[Dict],
        "definition_location": Optional[str],
        "last_modified_by": str,
        "last_modified_at": str
    }
    """
    try:
        node_id = request.path_params["node_id"]
        db = get_db()

        if not db.has_node(node_id):
            return error_response(f"Node not found: {node_id}", 404)

        node = db.get_node(node_id)

        # Get messages that reference this node
        message_nodes = db.get_messages_for_node(node_id)

        # Get incoming and outgoing edges
        incoming_edges = db.get_incoming_edges(node_id)
        outgoing_edges = db.get_outgoing_edges(node_id)

        # Extract metadata from node
        definition_location = None
        if hasattr(node, 'metadata') and node.metadata:
            extra = node.metadata.extra if hasattr(node.metadata, 'extra') else {}
            definition_location = extra.get('file_path') or extra.get('location')

        last_modified_by = node.created_by if hasattr(node, 'created_by') else "unknown"
        last_modified_at = node.created_at if hasattr(node, 'created_at') else ""

        # Serialize message nodes
        messages_data = [
            {
                "message_id": msg.id,
                "content": msg.content[:200] if msg.content else "",  # Truncate
                "created_at": msg.created_at,
                "created_by": msg.created_by,
            }
            for msg in message_nodes
        ]

        return JSONResponse({
            "node_id": node_id,
            "referenced_in_messages": messages_data,
            "incoming_edges": incoming_edges,
            "outgoing_edges": outgoing_edges,
            "definition_location": definition_location,
            "last_modified_by": last_modified_by,
            "last_modified_at": last_modified_at,
            "message_count": len(messages_data),
        })

    except Exception as e:
        logger.error(f"Error in get_node_reverse_connections: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


async def get_node_messages(request: Request) -> JSONResponse:
    """
    GET /api/nodes/{node_id}/messages

    Get all MESSAGE nodes that reference this node.
    Enhanced version with full message details.

    Response:
    {
        "node_id": str,
        "messages": List[{
            "message_id": str,
            "content": str,
            "created_at": str,
            "created_by": str,
            "message_type": str
        }],
        "count": int
    }
    """
    try:
        node_id = request.path_params["node_id"]
        db = get_db()

        if not db.has_node(node_id):
            return error_response(f"Node not found: {node_id}", 404)

        # Get messages that reference this node
        message_nodes = db.get_messages_for_node(node_id)

        # Get full message details
        messages = []
        for msg_node in message_nodes:
            messages.append({
                "message_id": msg_node.id,
                "content": msg_node.content if hasattr(msg_node, 'content') else "",
                "created_at": msg_node.created_at if hasattr(msg_node, 'created_at') else "",
                "created_by": msg_node.created_by if hasattr(msg_node, 'created_by') else "unknown",
                "message_type": msg_node.type if hasattr(msg_node, 'type') else "MESSAGE",
            })

        return JSONResponse({
            "node_id": node_id,
            "messages": messages,
            "count": len(messages),
        })

    except Exception as e:
        logger.error(f"Error in get_node_messages: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


# =============================================================================
# APP FACTORY
# =============================================================================

def create_routes() -> List[Route]:
    """Create all API routes."""
    return [
        # Health & Stats
        Route("/health", health, methods=["GET"]),
        Route("/stats", stats, methods=["GET"]),

        # Node operations
        Route("/nodes", create_node, methods=["POST"]),
        Route("/nodes", list_nodes, methods=["GET"]),
        Route("/nodes/{node_id}", get_node, methods=["GET"]),
        Route("/nodes/{node_id}/dialogue", get_node_dialogue, methods=["GET"]),
        Route("/api/research/nodes", get_research_nodes, methods=["GET"]),

        # Edge operations
        Route("/edges", create_edge, methods=["POST"]),
        Route("/edges", list_edges, methods=["GET"]),

        # Graph operations
        Route("/waves", get_waves, methods=["GET"]),
        Route("/descendants/{node_id}", get_descendants, methods=["GET"]),
        Route("/ancestors/{node_id}", get_ancestors, methods=["GET"]),

        # Parsing
        Route("/parse", parse_source, methods=["POST"]),

        # Alignment
        Route("/align", align_graphs, methods=["POST"]),

        # Graph validation
        Route("/api/validate", validate_graph_endpoint, methods=["GET"]),

        # Visualization endpoints
        Route("/api/viz/snapshot", viz_snapshot, methods=["GET"]),
        Route("/api/viz/stream", viz_stream, methods=["GET"]),
        Route("/api/viz/compare", viz_compare, methods=["GET"]),
        Route("/api/viz/snapshots", viz_list_snapshots, methods=["GET"]),
        Route("/api/viz/snapshots", viz_save_snapshot, methods=["POST"]),

        # Dialectic endpoints (Phase 3)
        Route("/api/dialector/questions", get_dialector_questions, methods=["GET"]),
        Route("/api/dialector/answer", submit_dialector_answer, methods=["POST"]),
        Route("/api/orchestrator/state", get_orchestrator_state, methods=["GET"]),

        # Orchestrator session management
        Route("/api/orchestrator/sessions", start_orchestrator_session, methods=["POST"]),
        Route("/api/orchestrator/current", get_current_session, methods=["GET"]),
        Route("/api/orchestrator/run", run_orchestrator_phase, methods=["POST"]),
        Route("/api/orchestrator/message", send_orchestrator_message, methods=["POST"]),
        Route("/api/orchestrator/conversation", get_conversation_history, methods=["GET"]),

        # Edge case tracking
        Route("/api/edge-cases", list_edge_cases, methods=["GET"]),
        Route("/api/edge-cases/summary", edge_case_summary, methods=["GET"]),
        Route("/api/edge-cases/flag", flag_edge_case, methods=["POST"]),
        Route("/api/edge-cases/{edge_case_id}", get_edge_case, methods=["GET"]),
        Route("/api/edge-cases/{edge_case_id}/resolve", resolve_edge_case, methods=["POST"]),

        # Speech-to-text endpoints
        Route("/api/stt/status", stt_status, methods=["GET"]),
        Route("/api/stt/transcribe", transcribe_audio_file, methods=["POST"]),

        # Parallel Pipeline Detection (Layer 8 Physics)
        Route("/api/physics/parallel-pipelines", get_parallel_pipeline_detections, methods=["GET"]),
        Route("/api/physics/parallel-pipelines/review", review_parallel_pipeline_detection, methods=["POST"]),
        Route("/api/physics/parallel-pipelines/whitelist", get_parallel_pipeline_whitelist, methods=["GET"]),

        # Dialogue-to-Graph Highlighting (Interactive Visualization)
        Route("/api/graph/highlight", graph_highlight, methods=["POST"]),
        Route("/api/nodes/{node_id}/reverse-connections", get_node_reverse_connections, methods=["GET"]),
        Route("/api/nodes/{node_id}/messages", get_node_messages, methods=["GET"]),

        # Notification endpoints (Cross-Tab Notifications)
        Route("/api/notifications/pending", get_pending_notifications, methods=["GET"]),
        Route("/api/notifications/{notification_id}/mark-read", mark_notification_read, methods=["POST"]),
        Route("/api/notifications/create", create_notification_endpoint, methods=["POST"]),
    ]


def create_websocket_routes() -> List[WebSocketRoute]:
    """Create WebSocket routes."""
    return [
        WebSocketRoute("/api/viz/ws", viz_websocket),
        WebSocketRoute("/api/dialectic/ws", dialectic_websocket),
        WebSocketRoute("/ws/audio", audio_streaming_websocket),
    ]


def create_app() -> Starlette:
    """Create the Starlette application."""
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware

    # CORS middleware for frontend access
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]

    all_routes = create_routes() + create_websocket_routes()
    app = Starlette(
        routes=all_routes,
        middleware=middleware,
        debug=False,
    )

    # Subscribe to graph events on startup
    if EVENT_BUS_AVAILABLE:
        event_bus = get_event_bus()
        event_bus.subscribe_async(EventType.NODE_CREATED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.NODE_UPDATED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.NODE_DELETED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.EDGE_CREATED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.EDGE_DELETED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.ORCHESTRATOR_ERROR, broadcast_graph_event)
        event_bus.subscribe_async(EventType.PHASE_CHANGED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.NOTIFICATION_CREATED, broadcast_graph_event)
        event_bus.subscribe_async(EventType.DIALOGUE_TURN_ADDED, broadcast_graph_event)
        logger.info("Subscribed to graph events for WebSocket broadcasting")

    return app


# Application instance for ASGI servers
app = create_app()
