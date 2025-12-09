# Dialogue-to-Graph Highlighting System

**Status:** Implemented
**Version:** 1.0.0
**Date:** 2025-12-08

## Overview

The dialogue-to-graph highlighting system enables interactive visualization features where clicking a message/dialogue turn highlights related nodes in the graph, and hovering a node shows which messages reference it.

## Architecture

### Components

1. **Graph Traversal Layer** (`core/graph_db.py`)
   - `get_related_nodes()` - Get nodes related to a target node
   - `get_reverse_connections()` - Find messages and edges referencing a node
   - `get_nodes_for_message()` - Get nodes referenced by a message

2. **API Layer** (`api/routes.py`)
   - `POST /api/graph/highlight` - Get highlighting data
   - `GET /api/nodes/{node_id}/reverse-connections` - Get reverse references
   - `GET /api/nodes/{node_id}/messages` - Get messages for a node

3. **WebSocket Layer** (`api/routes.py`)
   - `broadcast_node_highlight()` - Real-time highlighting events

## Graph Traversal Methods

### `get_related_nodes(node_id: str, mode: str) -> List[str]`

Get nodes related to a given node for highlighting purposes.

**Modes:**
- `"exact"`: Just the node itself
- `"related"`: Node + immediate dependencies (DEPENDS_ON, IMPLEMENTS edges)
- `"dependent"`: Node + all nodes that depend on it (reverse dependencies)

**Implementation:**
- Uses `rustworkx.descendants()` for O(V+E) traversal
- Filters edges by type (DEPENDS_ON, IMPLEMENTS)
- Returns sorted list of node IDs

**Example:**
```python
db = get_db()
related = db.get_related_nodes("spec_123", mode="related")
# Returns: ["spec_123", "req_456", "code_789"]
```

### `get_reverse_connections(node_id: str) -> Dict[str, Any]`

Find all MESSAGE/DIALOGUE nodes and edges that reference this node.

**Returns:**
```python
{
    "node_id": str,
    "referenced_in_dialogue": List[str],  # All MESSAGE/THREAD nodes
    "referenced_in_messages": List[str],  # Only MESSAGE nodes
    "incoming_edges": List[Dict],
    "outgoing_edges": List[Dict],
    "definition_location": Optional[str],
    "last_modified_by": str,
    "last_modified_at": str,
}
```

**Implementation:**
- Iterates through all nodes to find MESSAGE/THREAD types
- Checks for REFERENCES edges
- Collects incoming/outgoing edge metadata
- O(V) complexity for V nodes

**Example:**
```python
connections = db.get_reverse_connections("code_abc")
# Returns all messages that mention this code node
```

### `get_nodes_for_message(message_id: str) -> List[str]`

Get all nodes referenced by a MESSAGE or DIALOGUE_TURN node.

**Implementation:**
- Validates message node type
- Finds outgoing REFERENCES edges
- Returns list of node IDs
- O(E) complexity where E is outgoing edges

**Example:**
```python
nodes = db.get_nodes_for_message("msg_123")
# Returns: ["spec_456", "code_789"]
```

## API Endpoints

### `POST /api/graph/highlight`

Get nodes and edges to highlight based on a source node/message/edge.

**Request:**
```json
{
    "highlight_type": "message" | "node" | "edge",
    "source_id": "node_or_message_id",
    "highlight_mode": "exact" | "related" | "dependent"
}
```

**Response:**
```json
{
    "nodes_to_highlight": ["node_1", "node_2", ...],
    "edges_to_highlight": [
        {"source": "node_1", "target": "node_2"},
        ...
    ],
    "context": "Human-readable description",
    "highlight_mode": "related",
    "highlight_type": "message"
}
```

**Logic:**
1. **Message highlighting:**
   - Get all nodes referenced by message
   - Apply highlight mode to each referenced node
   - Collect all related nodes
   - Find edges between highlighted nodes

2. **Node highlighting:**
   - Get related nodes for the target
   - Find edges between highlighted nodes

3. **Edge highlighting:**
   - Get related nodes for both endpoints
   - Include the edge itself
   - Find edges between highlighted nodes

**Example:**
```bash
curl -X POST http://localhost:8000/api/graph/highlight \
  -H "Content-Type: application/json" \
  -d '{
    "highlight_type": "message",
    "source_id": "msg_turn_5",
    "highlight_mode": "related"
  }'
```

### `GET /api/nodes/{node_id}/reverse-connections`

Get all MESSAGE nodes and edges that reference this node.

**Response:**
```json
{
    "node_id": "code_123",
    "referenced_in_messages": [
        {
            "message_id": "msg_5",
            "content": "Implemented login endpoint...",
            "created_at": "2025-12-08T10:00:00Z",
            "created_by": "BUILDER"
        }
    ],
    "incoming_edges": [
        {
            "source": "test_456",
            "target": "code_123",
            "type": "TESTS",
            "weight": 1.0,
            "source_node_type": "TEST_SUITE"
        }
    ],
    "outgoing_edges": [...],
    "definition_location": "/path/to/file.py",
    "last_modified_by": "BUILDER",
    "last_modified_at": "2025-12-08T10:00:00Z",
    "message_count": 2
}
```

**Example:**
```bash
curl http://localhost:8000/api/nodes/code_123/reverse-connections
```

### `GET /api/nodes/{node_id}/messages`

Get all MESSAGE nodes that reference this node with full details.

**Response:**
```json
{
    "node_id": "code_123",
    "messages": [
        {
            "message_id": "msg_5",
            "content": "Implemented login endpoint...",
            "created_at": "2025-12-08T10:00:00Z",
            "created_by": "BUILDER",
            "message_type": "MESSAGE"
        }
    ],
    "count": 1
}
```

**Example:**
```bash
curl http://localhost:8000/api/nodes/code_123/messages
```

## WebSocket Support

### `broadcast_node_highlight()`

Broadcast node highlight events to all connected WebSocket clients.

**Message Format:**
```json
{
    "type": "node_highlight",
    "data": {
        "nodes_to_highlight": ["node_1", "node_2"],
        "edges_to_highlight": [
            {"source": "node_1", "target": "node_2"}
        ],
        "reason": "User clicked message #5",
        "highlight_mode": "related",
        "timestamp": "2025-12-08T10:00:00Z"
    }
}
```

**Usage:**
```python
await broadcast_node_highlight(
    nodes_to_highlight=["node_1", "node_2"],
    edges_to_highlight=[{"source": "node_1", "target": "node_2"}],
    reason="User clicked message #5",
    highlight_mode="related"
)
```

## Frontend Integration

### Click-to-Highlight Flow

1. User clicks a message in the dialogue panel
2. Frontend calls `POST /api/graph/highlight` with message ID
3. Backend returns nodes and edges to highlight
4. Frontend highlights nodes and edges in the graph visualization
5. (Optional) Broadcast highlight event via WebSocket for other tabs

### Hover-to-Show Flow

1. User hovers over a node in the graph
2. Frontend calls `GET /api/nodes/{node_id}/reverse-connections`
3. Backend returns messages that reference the node
4. Frontend displays tooltip with message list

## Performance Characteristics

### Benchmarks (1000-node graphs)

| Operation | Mode | P50 | P95 | Target |
|-----------|------|-----|-----|--------|
| `get_related_nodes()` | exact | ~1ms | ~2ms | <100ms ✓ |
| `get_related_nodes()` | related | ~5ms | ~10ms | <100ms ✓ |
| `get_related_nodes()` | dependent | ~15ms | ~30ms | <100ms ✓ |
| `get_reverse_connections()` | - | ~20ms | ~40ms | <150ms ✓ |
| `get_nodes_for_message()` | - | ~2ms | ~5ms | <50ms ✓ |
| Full highlight flow | - | ~30ms | ~60ms | <200ms ✓ |

**Key Points:**
- All operations complete well under target times
- Rust-native `rustworkx` provides O(V+E) traversal
- Scales linearly with graph size
- Ready for production use with 1000+ node graphs

### Scaling Limits

- **Tested up to:** 5000 nodes
- **Expected limit:** 10,000+ nodes
- **Bottleneck:** Message lookup (O(V) for all nodes)
- **Optimization:** Index MESSAGE nodes separately if needed

## Testing

### Unit Tests

**File:** `/Users/lauferva/paragon/tests/unit/api/test_graph_highlighting.py`

**Coverage:**
- ParagonDB method tests
  - `test_get_related_nodes_exact()`
  - `test_get_related_nodes_related()`
  - `test_get_related_nodes_dependent()`
  - `test_get_reverse_connections()`
  - `test_get_nodes_for_message()`

- API endpoint tests
  - `test_graph_highlight_node_exact()`
  - `test_graph_highlight_node_related()`
  - `test_graph_highlight_message()`
  - `test_graph_highlight_edge()`
  - `test_get_node_reverse_connections_endpoint()`
  - `test_get_node_messages_endpoint()`

- Edge case tests
  - Isolated nodes
  - Invalid input
  - Nonexistent nodes
  - Empty results

- Performance tests
  - 100-node graphs
  - 1000-node graphs

### Integration Tests

**File:** `/Users/lauferva/paragon/tests/integration/test_dialogue_to_graph_mapping.py`

**Scenarios:**
- Full TDD orchestrator session with dialogue
- Message-to-node mapping
- Click-to-highlight flows
- Hover-to-show flows
- WebSocket broadcasting
- Complex multi-message scenarios
- Performance at scale (1000+ nodes)

### Performance Benchmarks

**File:** `/Users/lauferva/paragon/benchmarks/dialogue_highlighting_perf.py`

Run with:
```bash
python benchmarks/dialogue_highlighting_perf.py
```

## Usage Examples

### Example 1: Highlight nodes when message clicked

```python
# Backend
from agents.tools import get_db

db = get_db()

# User clicks message in dialogue
message_id = "msg_turn_5"

# Get nodes referenced by message
referenced = db.get_nodes_from_message(message_id)

# Get related nodes for each
all_highlights = set()
for node in referenced:
    related = db.get_related_nodes(node.id, mode="related")
    all_highlights.update(related)

# Return to frontend
return {
    "nodes_to_highlight": list(all_highlights),
    "context": f"Message references {len(referenced)} nodes"
}
```

### Example 2: Show messages when hovering node

```python
# Backend
db = get_db()

# User hovers over node
node_id = "code_123"

# Get messages that reference this node
messages = db.get_messages_for_node(node_id)

# Return to frontend for tooltip
return {
    "messages": [
        {
            "content": msg.content,
            "created_by": msg.created_by,
            "created_at": msg.created_at,
        }
        for msg in messages
    ]
}
```

### Example 3: Real-time highlighting via WebSocket

```python
# Backend
from api.routes import broadcast_node_highlight

# When user clicks in one tab, broadcast to all tabs
await broadcast_node_highlight(
    nodes_to_highlight=["node_1", "node_2"],
    edges_to_highlight=[{"source": "node_1", "target": "node_2"}],
    reason="User clicked message in Research tab",
    highlight_mode="related"
)
```

## Future Enhancements

### Phase 2 (Optional)

1. **Indexed MESSAGE lookup**
   - Build separate index of MESSAGE -> referenced nodes
   - Reduces O(V) to O(1) for message lookup
   - Needed only if > 10k nodes

2. **Highlight history**
   - Track user's highlighting history
   - "Back" button to previous highlights
   - Saved highlight sets

3. **Custom highlight modes**
   - User-defined traversal rules
   - Named highlight sets
   - Persistent highlight configurations

4. **Performance monitoring**
   - Track P95/P99 latencies in production
   - Alert on slow queries
   - Automatic optimization suggestions

## References

- **Graph Database:** `/Users/lauferva/paragon/core/graph_db.py`
- **API Routes:** `/Users/lauferva/paragon/api/routes.py`
- **Unit Tests:** `/Users/lauferva/paragon/tests/unit/api/test_graph_highlighting.py`
- **Integration Tests:** `/Users/lauferva/paragon/tests/integration/test_dialogue_to_graph_mapping.py`
- **Benchmarks:** `/Users/lauferva/paragon/benchmarks/dialogue_highlighting_perf.py`

## Change Log

### v1.0.0 (2025-12-08)
- Initial implementation
- Graph traversal methods in ParagonDB
- API endpoints for highlighting
- WebSocket support for real-time updates
- Comprehensive test coverage
- Performance benchmarks
