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
        return json_response({"created": len(nodes), "ids": [n.id for n in nodes]}, status_code=201)
    else:
        node = NodeData.create(
            type=body.get("type", NodeType.CODE.value),
            content=body.get("content", ""),
            data=body.get("data", {}),
            created_by=body.get("created_by", "api"),
        )
        db.add_node(node)
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
    ]


def create_websocket_routes() -> List[WebSocketRoute]:
    """Create WebSocket routes."""
    return [
        WebSocketRoute("/api/viz/ws", viz_websocket),
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
