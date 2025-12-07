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


# =============================================================================
# GLOBAL STATE
# =============================================================================

# Single database instance (will be initialized in create_app)
_db: Optional[ParagonDB] = None
_parser: Optional[CodeParser] = None

# WebSocket connections for real-time updates
_ws_connections: Set[WebSocket] = set()

# Snapshot cache for comparison view
_snapshot_cache: Dict[str, GraphSnapshot] = {}

# Sequence counter for delta messages
_sequence_counter: int = 0

# Orchestrator instance for dialectic interactions
_orchestrator_instance: Optional[Any] = None
_orchestrator_state: Dict[str, Any] = {}


def _next_sequence() -> int:
    """Get next sequence number for delta messages."""
    global _sequence_counter
    _sequence_counter += 1
    return _sequence_counter


def get_db() -> ParagonDB:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = ParagonDB()
    return _db


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

    # Handle batch vs single
    if isinstance(body, list):
        nodes = []
        for item in body:
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
            "phase": "clarification",
            "session_id": "session_id"
        }
    """
    global _orchestrator_state

    # Check if we have an active orchestrator state with pending questions
    questions = _orchestrator_state.get("clarification_questions", [])
    phase = _orchestrator_state.get("phase", "init")
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

    session_id = body.get("session_id")
    answers = body.get("answers", [])

    if not session_id:
        return error_response("session_id is required", status_code=400)

    if not answers:
        return error_response("answers array is required", status_code=400)

    # Format answers as a human response
    answer_text = "\n".join([
        f"{i+1}. {ans.get('answer', '')}"
        for i, ans in enumerate(answers)
    ])

    global _orchestrator_state
    _orchestrator_state["human_response"] = answer_text
    _orchestrator_state["pending_human_input"] = None

    # Try to resume orchestrator if available
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
            "phase": "clarification",
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
                    "phase": state.get("phase", "init"),
                    "session_id": session_id,
                    "has_pending_input": state.get("pending_human_input") is not None,
                    "iteration": state.get("iteration", 0),
                    "dialectic_passed": state.get("dialectic_passed", False),
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
        "dialectic_passed": _orchestrator_state.get("dialectic_passed", False),
        "research_complete": _orchestrator_state.get("research_complete", False),
    })


# =============================================================================
# DIALECTIC WEBSOCKET (Phase 3 - Real-time Updates)
# =============================================================================

# WebSocket connections for dialectic updates (session_id -> websocket)
_dialectic_ws_connections: Dict[str, WebSocket] = {}


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
        initial_state = {
            "type": "state_update",
            "data": {
                "phase": _orchestrator_state.get("phase", "init"),
                "session_id": session_id or _orchestrator_state.get("session_id", ""),
                "has_questions": len(_orchestrator_state.get("clarification_questions", [])) > 0,
                "questions": _format_questions_for_ws(_orchestrator_state.get("clarification_questions", [])),
                "dialectic_passed": _orchestrator_state.get("dialectic_passed", False),
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
                    await websocket.send_json({
                        "type": "state_update",
                        "data": {
                            "phase": _orchestrator_state.get("phase", "research"),
                            "session_id": session_id,
                            "has_questions": False,
                            "dialectic_passed": True,
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
    """Format clarification questions for WebSocket transmission."""
    formatted = []
    for i, q in enumerate(questions):
        formatted.append({
            "id": f"q_{i}",
            "question": q.get("question", ""),
            "category": q.get("category", "MISSING_CONTEXT"),
            "text": q.get("text", ""),
            "suggested_answer": q.get("suggested_answer"),
        })
    return formatted


async def broadcast_dialectic_update(session_id: str, update: Dict[str, Any]) -> None:
    """Broadcast dialectic state update to connected client."""
    if session_id in _dialectic_ws_connections:
        ws = _dialectic_ws_connections[session_id]
        try:
            await ws.send_json({
                "type": "state_update",
                "data": update
            })
        except Exception:
            del _dialectic_ws_connections[session_id]


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
            "phase": "dialectic"
        }
    """
    try:
        body = await request.json()
    except Exception as e:
        return error_response(f"Invalid JSON: {e}")

    requirement_text = body.get("requirement_text", "")
    requirements = body.get("requirements", [])

    if not requirement_text:
        return error_response("requirement_text is required", status_code=400)

    # Generate session ID
    session_id = f"session_{_uuid.uuid4().hex[:12]}"
    task_id = f"task_{_uuid.uuid4().hex[:8]}"

    # Update global state for polling endpoints
    global _orchestrator_state
    _orchestrator_state = {
        "session_id": session_id,
        "task_id": task_id,
        "phase": "dialectic",
        "spec": requirement_text,
        "requirements": requirements,
        "clarification_questions": [],
        "dialectic_passed": False,
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
        "phase": "dialectic",
        "orchestrator_available": _orchestrator_state.get("orchestrator_available", False),
    }, status_code=201)


async def run_orchestrator_phase(request: Request) -> JSONResponse:
    """
    Run the next phase of an orchestrator session.

    Body:
        {
            "session_id": "...",
            "phase": "dialectic"  # optional - run specific phase
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
            "dialectic_passed": result.get("dialectic_passed", False),
            "final_status": result.get("final_status"),
        })

    except Exception as e:
        return error_response(f"Orchestrator error: {e}", status_code=500)


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
        Route("/api/orchestrator/run", run_orchestrator_phase, methods=["POST"]),
    ]


def create_websocket_routes() -> List[WebSocketRoute]:
    """Create WebSocket routes."""
    return [
        WebSocketRoute("/api/viz/ws", viz_websocket),
        WebSocketRoute("/api/dialectic/ws", dialectic_websocket),
    ]


def create_app() -> Starlette:
    """Create the Starlette application."""
    all_routes = create_routes() + create_websocket_routes()
    return Starlette(
        routes=all_routes,
        debug=False,
    )


# Application instance for ASGI servers
app = create_app()
