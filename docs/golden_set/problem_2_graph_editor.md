# GOLDEN SET PROBLEM 2: Real-time Collaborative Graph Editor

**Complexity Level:** Very High
**Category:** Real-time Systems, Distributed Consensus, Conflict Resolution, Frontend + Backend
**Estimated Components:** 20-25
**Test Coverage Target:** 100% (production quality gate)

---

## EXECUTIVE SUMMARY

Build a collaborative graph editor where multiple users can simultaneously edit a directed acyclic graph (DAG) with real-time synchronization, operational transformation for conflict resolution, undo/redo with multi-user awareness, WebSocket-based communication, and graph constraint validation. Think "Google Docs for graphs" with Paragon-style invariant enforcement.

This problem tests:
- Real-time collaboration (Operational Transformation, CRDT-style conflict resolution)
- WebSocket protocol design (client-server bidirectional communication)
- Distributed state management (eventual consistency, vector clocks)
- Complex undo/redo (collaborative history, causal ordering)
- Graph constraint enforcement (DAG preservation, custom validation rules)
- Frontend + Backend orchestration (React + FastAPI)
- Performance under concurrent load (100+ users editing same graph)

---

## 1. FUNCTIONAL REQUIREMENTS

### FR-1: Graph Data Model
**Priority:** P0 (Blocking)

The system MUST represent graphs with full CRUD operations:
- **Nodes:** Create, update, delete with properties (type, label, position, data)
- **Edges:** Create, delete with validation (no self-loops, no cycles)
- **Graph:** Immutable snapshots for undo/redo

**Schema:**
```python
class GraphNode(msgspec.Struct, kw_only=True):
    id: str  # UUID
    type: str  # User-defined types: "task", "decision", "data", etc.
    label: str
    position: Position  # x, y coordinates for visual layout
    data: Dict[str, Any]  # Arbitrary user data (JSON-serializable)
    created_by: str  # User ID
    created_at: str  # ISO 8601 timestamp
    version: int  # Incremented on each edit (for OT)

class Position(msgspec.Struct, kw_only=True):
    x: float
    y: float

class GraphEdge(msgspec.Struct, kw_only=True):
    id: str
    source_id: str
    target_id: str
    edge_type: str  # "solid", "dashed", "arrow", etc.
    label: Optional[str] = None
    created_by: str
    created_at: str
    version: int

class GraphSnapshot(msgspec.Struct, kw_only=True):
    snapshot_id: str
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    created_at: str
    created_by_operation: str  # Operation that created this snapshot
```

**Constraints:**
1. **DAG Enforcement:** No cycles allowed (detect via DFS after edge creation)
2. **No Dangling Edges:** Deleting node must delete incident edges
3. **Unique IDs:** Node/edge IDs must be globally unique (UUIDs)
4. **Type Safety:** Node types must be from predefined schema (extendable)
5. **Position Validity:** x, y coordinates must be finite numbers

**Edge Cases:**
1. Simultaneous node deletion by two users → Conflict resolution
2. Add edge that would create cycle → Reject with error
3. Edge referencing non-existent node → Reject
4. Position update to same coordinates → Deduplicate
5. Massive graph (10,000+ nodes) → Pagination, lazy loading

**Test Scenarios:**
- Create graph with 100 nodes, verify all constraints hold
- Attempt to create cycle, verify rejection
- Delete node, verify edges also deleted
- Property-based: Any sequence of valid ops maintains DAG

---

### FR-2: WebSocket Communication Protocol
**Priority:** P0 (Blocking)

The system MUST use WebSocket for bidirectional real-time communication:
- **Client → Server:** User operations (create node, move node, etc.)
- **Server → Client:** Broadcasts (other users' operations, acknowledgments)
- **Heartbeat:** Ping/pong every 30s to detect disconnections
- **Reconnection:** Automatic reconnect with state sync

**Message Protocol:**
```python
class WSMessage(msgspec.Struct, kw_only=True):
    message_id: str  # UUID for deduplication
    session_id: str  # Graph editing session
    user_id: str
    message_type: Literal[
        "operation",      # User operation (add node, etc.)
        "broadcast",      # Server broadcast to all clients
        "ack",           # Acknowledgment of operation
        "snapshot",      # Full graph snapshot
        "heartbeat",     # Ping/pong
        "error"          # Error response
    ]
    payload: Dict[str, Any]
    timestamp: str  # ISO 8601
    vector_clock: Dict[str, int]  # For causal ordering

class Operation(msgspec.Struct, kw_only=True):
    op_id: str  # Unique operation ID
    op_type: Literal[
        "add_node", "update_node", "delete_node",
        "add_edge", "delete_edge",
        "move_node"  # Update position only
    ]
    user_id: str
    session_id: str
    data: Dict[str, Any]  # Operation-specific data
    dependencies: List[str] = []  # IDs of operations this depends on (for OT)
    vector_clock: Dict[str, int]  # Logical timestamp
```

**Workflow:**
1. Client performs edit → Generate Operation
2. Send Operation via WebSocket to server
3. Server validates Operation (constraints, permissions)
4. Server transforms Operation (resolve conflicts with concurrent ops)
5. Server applies Operation to graph
6. Server broadcasts transformed Operation to all clients
7. Clients apply broadcast Operation to local state

**Edge Cases:**
1. Message arrives out of order → Use vector clock to reorder
2. Duplicate message (retry) → Deduplicate by message_id
3. Disconnection during operation → Buffer on client, resend on reconnect
4. Server restart → Clients reconnect, request full snapshot
5. Network partition → Queue operations, sync when partition heals

**Test Scenarios:**
- Mock WebSocket: Send operation, verify broadcast
- Simulate out-of-order delivery, verify reordering
- Simulate disconnect, verify reconnect and sync
- Load test: 100 clients send operations simultaneously

---

### FR-3: Operational Transformation (OT)
**Priority:** P0 (Blocking)

The system MUST use OT to resolve conflicting concurrent operations:
- **Transform Function:** `transform(op1, op2) → (op1', op2')`
- **Convergence:** All clients converge to same graph state
- **Causality Preservation:** Operations respect causal dependencies
- **Intention Preservation:** User intent is maintained after transformation

**OT Algorithm:**
```python
def transform(op1: Operation, op2: Operation) -> Tuple[Operation, Operation]:
    """
    Transform two concurrent operations to preserve intent.

    Returns (op1', op2') where:
    - Applying op1 then op2' = Applying op2 then op1'

    Examples:
    - Both users add node at same position → Offset one by (10, 10)
    - User A deletes node, User B adds edge to it → Discard B's edge
    - Both users update same node label → Last-writer-wins (timestamp)
    """
    if op1.op_type == "delete_node" and op2.op_type == "add_edge":
        if op2.data["source_id"] == op1.data["node_id"] or \
           op2.data["target_id"] == op1.data["node_id"]:
            # Node deleted, discard edge creation
            return (op1, noop_operation())

    if op1.op_type == "add_node" and op2.op_type == "add_node":
        if op1.data["position"] == op2.data["position"]:
            # Same position, offset second node
            op2_prime = copy(op2)
            op2_prime.data["position"]["x"] += 10
            op2_prime.data["position"]["y"] += 10
            return (op1, op2_prime)

    if op1.op_type == "update_node" and op2.op_type == "update_node":
        if op1.data["node_id"] == op2.data["node_id"]:
            # Same node updated, last-writer-wins
            if op1.timestamp > op2.timestamp:
                return (op1, noop_operation())
            else:
                return (noop_operation(), op2)

    # Default: operations are independent
    return (op1, op2)
```

**Transformation Rules:**
1. **Delete + Any:** Delete takes precedence (node is gone)
2. **Position Conflict:** Offset by small delta
3. **Property Update Conflict:** Last-writer-wins (timestamp)
4. **Cycle Creation:** Reject edge addition
5. **Independent Ops:** No transformation needed

**Edge Cases:**
1. Three-way conflict (A, B, C edit simultaneously) → Pairwise transform
2. Long operation chain (10+ dependent ops) → Optimize with OT tree
3. Conflicting delete (both delete same node) → Idempotent, both succeed
4. Transform creates invalid state (cycle) → Reject original op

**Test Scenarios:**
- Property-based: transform(op1, op2) must converge
- Conflict scenarios: All pairwise conflicts tested
- Convergence test: 100 random operations, all clients reach same state
- Correctness: Apply ops in different orders, verify same result

---

### FR-4: Undo/Redo with Multi-User Awareness
**Priority:** P1 (High)

The system MUST support undo/redo that respects multi-user context:
- **Local Undo:** User can undo their own operations
- **Cascading Undo:** If undoing op A invalidates op B (by another user), notify B
- **Redo:** Re-apply undone operation with transformation
- **History Visualization:** Show who did what, when

**Undo Strategy:**
```python
class UndoStack(msgspec.Struct, kw_only=True):
    user_id: str
    operations: List[Operation]  # Chronological order
    undo_index: int  # Current position in stack

def undo(user_id: str) -> Operation:
    """
    Undo the last operation by user_id.

    Returns the inverse operation to apply.
    """
    stack = get_undo_stack(user_id)
    if stack.undo_index == 0:
        raise ValueError("Nothing to undo")

    op = stack.operations[stack.undo_index - 1]
    inverse_op = create_inverse_operation(op)

    # Check if undoing this op invalidates others
    dependent_ops = find_dependent_operations(op)
    if dependent_ops:
        notify_users_of_invalidation(dependent_ops)

    stack.undo_index -= 1
    return inverse_op

def create_inverse_operation(op: Operation) -> Operation:
    """Generate inverse operation to undo."""
    if op.op_type == "add_node":
        return Operation(op_type="delete_node", data={"node_id": op.data["id"]})
    elif op.op_type == "delete_node":
        return Operation(op_type="add_node", data=op.data["node"])
    elif op.op_type == "update_node":
        return Operation(op_type="update_node", data={"node_id": op.data["node_id"], "previous_value": op.data["value"]})
    # ... etc.
```

**Dependency Detection:**
```python
def find_dependent_operations(op: Operation) -> List[Operation]:
    """
    Find operations that depend on op.

    Example: If op adds node N, and later op2 adds edge to N,
    then undoing op should notify user who created op2.
    """
    if op.op_type == "add_node":
        node_id = op.data["id"]
        # Find edges referencing this node
        dependent_edges = [
            edge_op for edge_op in all_operations
            if edge_op.op_type == "add_edge" and
               (edge_op.data["source_id"] == node_id or edge_op.data["target_id"] == node_id)
        ]
        return dependent_edges
    return []
```

**Edge Cases:**
1. Undo node creation → Delete edges created by others referencing it
2. Undo while others are editing → Transform undo operation
3. Redo after other edits → Re-transform with new state
4. Undo cascade (A → B → C) → Notify all affected users
5. Undo limit (max 100 operations) → Discard old history

**Test Scenarios:**
- User creates node, adds edge, undoes node → Edge also deleted
- User A creates node, User B adds edge to it, User A undoes → B notified
- Undo/redo 100 times, verify state consistency
- Concurrent undo by multiple users, verify convergence

---

### FR-5: Graph Constraint Validation
**Priority:** P0 (Blocking)

The system MUST enforce customizable graph constraints:
- **DAG Constraint:** No cycles (detect via topological sort)
- **Degree Constraint:** Max in-degree, max out-degree per node
- **Type Constraint:** Only certain edge types between certain node types
- **Custom Constraints:** User-defined validation functions (Python expressions)

**Constraint Schema:**
```python
class GraphConstraint(msgspec.Struct, kw_only=True):
    constraint_id: str
    name: str
    description: str
    constraint_type: Literal["dag", "degree", "type", "custom"]
    config: Dict[str, Any]
    enabled: bool = True

class DegreeConstraint(msgspec.Struct, kw_only=True):
    max_in_degree: Optional[int] = None
    max_out_degree: Optional[int] = None
    applies_to_types: List[str] = []  # Empty = all types

class TypeConstraint(msgspec.Struct, kw_only=True):
    source_type: str
    edge_type: str
    target_type: str
    allowed: bool  # True = allow, False = forbid
```

**Validation Algorithm:**
```python
def validate_operation(op: Operation, graph: GraphSnapshot) -> ValidationResult:
    """
    Validate operation against all enabled constraints.

    Returns ValidationResult with pass/fail and errors.
    """
    errors = []

    if op.op_type == "add_edge":
        source_id = op.data["source_id"]
        target_id = op.data["target_id"]

        # DAG constraint: Check for cycle
        temp_graph = graph.copy()
        temp_graph.add_edge(source_id, target_id)
        if has_cycle(temp_graph):
            errors.append("Adding this edge would create a cycle")

        # Degree constraint
        target_in_degree = len(temp_graph.get_incoming_edges(target_id))
        constraint = get_constraint("degree")
        if constraint.max_in_degree and target_in_degree > constraint.max_in_degree:
            errors.append(f"Target node in-degree exceeds max ({constraint.max_in_degree})")

        # Type constraint
        source_node = graph.get_node(source_id)
        target_node = graph.get_node(target_id)
        if not is_edge_allowed(source_node.type, op.data["edge_type"], target_node.type):
            errors.append(f"Edge type '{op.data['edge_type']}' not allowed between {source_node.type} and {target_node.type}")

    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

**Edge Cases:**
1. Complex cycle detection (1000+ nodes) → Use cached topological order
2. Constraint conflicts (degree allows, type forbids) → Aggregate errors
3. Constraint evaluation timeout (custom function too slow) → Abort after 1s
4. Dynamic constraint update (admin changes rules) → Re-validate all edges

**Test Scenarios:**
- Create edge that would create cycle, verify rejection
- Add edges until degree limit, verify next rejected
- Test all type constraint permutations
- Performance: Validate 1000-edge graph in <100ms

---

### FR-6: Session Management and Presence
**Priority:** P1 (High)

The system MUST track user presence and session state:
- **Active Users:** List of users currently editing
- **Cursor Tracking:** Show each user's cursor position
- **Selection Tracking:** Highlight nodes/edges being edited
- **User Colors:** Assign unique color to each user
- **Idle Detection:** Timeout after 5 min of inactivity

**Presence Schema:**
```python
class UserPresence(msgspec.Struct, kw_only=True):
    user_id: str
    username: str
    color: str  # Hex color code
    cursor_position: Optional[Position] = None
    selected_nodes: List[str] = []
    selected_edges: List[str] = []
    last_active: str  # ISO 8601 timestamp
    status: Literal["active", "idle", "disconnected"]

class SessionState(msgspec.Struct, kw_only=True):
    session_id: str
    graph_id: str
    active_users: List[UserPresence]
    created_at: str
    last_modified: str
```

**Presence Protocol:**
```python
# Client sends periodic presence updates
{
    "message_type": "presence",
    "user_id": "alice",
    "cursor_position": {"x": 100, "y": 200},
    "selected_nodes": ["node_1", "node_2"]
}

# Server broadcasts to all clients
{
    "message_type": "broadcast",
    "payload": {
        "type": "presence_update",
        "user": {
            "user_id": "alice",
            "color": "#FF5733",
            "cursor_position": {"x": 100, "y": 200},
            "selected_nodes": ["node_1", "node_2"]
        }
    }
}
```

**Edge Cases:**
1. User disconnects ungracefully → Detect via heartbeat timeout
2. Reconnect with different session ID → Merge presence
3. 100+ active users → Throttle presence updates (max 10/sec)
4. Cursor spam (malicious rapid updates) → Rate limit per user

**Test Scenarios:**
- Connect 10 users, verify all see each other
- Disconnect user, verify others notified within 30s
- Update cursor 1000 times/sec, verify rate limiting
- Test presence broadcast latency (<100ms)

---

### FR-7: State Synchronization and Conflict Resolution
**Priority:** P0 (Blocking)

The system MUST maintain eventual consistency:
- **Vector Clocks:** Track causal ordering of operations
- **Snapshot Synchronization:** Periodic full-state sync
- **Delta Synchronization:** Send only changes since last sync
- **Conflict Detection:** Identify divergent client states

**Vector Clock:**
```python
class VectorClock:
    """
    Logical clock for tracking causal ordering.

    clock[user_id] = number of operations by that user
    """
    def __init__(self):
        self.clock: Dict[str, int] = {}

    def increment(self, user_id: str):
        self.clock[user_id] = self.clock.get(user_id, 0) + 1

    def merge(self, other: 'VectorClock'):
        for user_id, count in other.clock.items():
            self.clock[user_id] = max(self.clock.get(user_id, 0), count)

    def is_causally_before(self, other: 'VectorClock') -> bool:
        """True if self happened before other."""
        return all(
            self.clock.get(uid, 0) <= other.clock.get(uid, 0)
            for uid in set(self.clock.keys()) | set(other.clock.keys())
        )
```

**Synchronization Algorithm:**
```python
def sync_client(client_id: str, client_vector_clock: VectorClock):
    """
    Sync client to latest server state.

    1. Compare client vector clock with server clock
    2. If client is behind, send missing operations
    3. If client has unseen operations, apply them
    4. If clocks are concurrent (conflict), resolve via OT
    """
    server_clock = get_server_vector_clock()

    if client_vector_clock.is_causally_before(server_clock):
        # Client is behind, send delta
        missing_ops = get_operations_since(client_vector_clock)
        send_to_client({"type": "sync_delta", "operations": missing_ops})
    elif server_clock.is_causally_before(client_vector_clock):
        # Client is ahead (shouldn't happen), request full state
        send_to_client({"type": "sync_snapshot", "graph": get_current_graph()})
    else:
        # Concurrent (conflict), resolve
        resolve_divergence(client_id, client_vector_clock)
```

**Edge Cases:**
1. Long disconnection (client misses 100+ ops) → Send snapshot instead of delta
2. Clock overflow (very long session) → Compact vector clock
3. Byzantine client (sends invalid clock) → Reject and force snapshot
4. Network partition (split-brain) → Detect via quorum, reject minority

**Test Scenarios:**
- Disconnect client, perform 50 ops, reconnect → Verify sync
- Concurrent operations from 10 clients → All converge
- Simulate network partition, verify detection
- Benchmark: Sync 1000-op delta in <1s

---

### FR-8: Frontend UI (React + Canvas)
**Priority:** P1 (High)

The system MUST provide a rich graph editing UI:
- **Canvas Rendering:** HTML5 Canvas or SVG for graph visualization
- **Drag-and-Drop:** Move nodes by dragging
- **Zoom and Pan:** Mouse wheel zoom, click-drag pan
- **Selection:** Click to select node/edge, Shift+click for multi-select
- **Context Menu:** Right-click for actions (delete, edit properties)
- **Minimap:** Overview of entire graph
- **Toolbar:** Add node, add edge, undo, redo buttons

**Component Architecture:**
```typescript
// Main canvas component
interface GraphCanvasProps {
    graph: GraphSnapshot
    localUser: User
    activeUsers: UserPresence[]
    onOperation: (op: Operation) => void
}

function GraphCanvas({ graph, localUser, activeUsers, onOperation }: GraphCanvasProps) {
    // State management (Zustand store)
    const { selectedNodes, selectedEdges, zoom, pan } = useGraphStore()

    // WebSocket connection
    const { sendOperation, subscribe } = useGraphWebSocket()

    // Rendering
    return (
        <Canvas
            width={window.innerWidth}
            height={window.innerHeight}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
        >
            {/* Render nodes */}
            {graph.nodes.map(node => (
                <Node
                    key={node.id}
                    node={node}
                    selected={selectedNodes.includes(node.id)}
                    onDragStart={() => handleNodeDragStart(node)}
                    onDrag={(newPos) => handleNodeDrag(node, newPos)}
                />
            ))}

            {/* Render edges */}
            {graph.edges.map(edge => (
                <Edge key={edge.id} edge={edge} />
            ))}

            {/* Render cursors */}
            {activeUsers.map(user => (
                <Cursor
                    key={user.user_id}
                    position={user.cursor_position}
                    color={user.color}
                    label={user.username}
                />
            ))}
        </Canvas>
    )
}
```

**UI State Management:**
```typescript
interface GraphUIState {
    // Selection
    selectedNodes: string[]
    selectedEdges: string[]

    // Viewport
    zoom: number  // 0.1 to 5.0
    pan: { x: number; y: number }

    // Editing
    isDragging: boolean
    dragStart: { x: number; y: number } | null
    isAddingEdge: boolean
    edgeSource: string | null

    // Modals
    showNodeProperties: boolean
    showSettings: boolean
}
```

**Edge Cases:**
1. Very large graph (1000+ nodes) → Use virtual scrolling, only render visible nodes
2. High zoom (10x) → Maintain crisp rendering with SVG scaling
3. Rapid panning → Debounce render updates (max 60 FPS)
4. Mobile touch → Pinch-to-zoom, two-finger pan

**Test Scenarios:**
- Render 100-node graph, verify performance (60 FPS)
- Drag node, verify position update broadcast
- Select multiple nodes, delete, verify batch operation
- Zoom/pan stress test: Rapid interactions should not lag

---

### FR-9: Conflict Notification and Resolution UI
**Priority:** P2 (Medium)

The system MUST notify users of conflicts and allow resolution:
- **Conflict Toast:** "User B deleted the node you were editing"
- **Conflict Dialog:** Show conflicting changes, allow user to choose
- **Auto-Resolution:** Prefer automatic resolution (OT), ask only when ambiguous
- **Conflict Log:** History of all conflicts and how they were resolved

**Conflict Types:**
1. **Edit-Delete:** User edits node that was deleted by another user
2. **Concurrent Edit:** Both users edit same property with different values
3. **Dependency Broken:** User deletes node that another's edge depends on
4. **Cycle Created:** User's edge combined with another's creates cycle

**Resolution UI:**
```typescript
interface ConflictDialogProps {
    conflict: Conflict
    onResolve: (resolution: Resolution) => void
}

interface Conflict {
    conflict_id: string
    type: "edit_delete" | "concurrent_edit" | "dependency_broken" | "cycle_created"
    your_operation: Operation
    their_operation: Operation
    conflicting_user: string
}

interface Resolution {
    choice: "yours" | "theirs" | "merge" | "cancel"
    merged_value?: any  // If choice is "merge"
}

function ConflictDialog({ conflict, onResolve }: ConflictDialogProps) {
    return (
        <Dialog>
            <h2>Conflict Detected</h2>
            <p>
                Your change: {describe_operation(conflict.your_operation)}
                <br />
                {conflict.conflicting_user}'s change: {describe_operation(conflict.their_operation)}
            </p>

            <div className="conflict-options">
                <button onClick={() => onResolve({ choice: "yours" })}>
                    Keep My Change
                </button>
                <button onClick={() => onResolve({ choice: "theirs" })}>
                    Accept Their Change
                </button>
                {conflict.type === "concurrent_edit" && (
                    <button onClick={() => onResolve({ choice: "merge" })}>
                        Merge Changes
                    </button>
                )}
                <button onClick={() => onResolve({ choice: "cancel" })}>
                    Cancel My Change
                </button>
            </div>
        </Dialog>
    )
}
```

**Edge Cases:**
1. Multiple conflicts simultaneously → Queue and resolve one by one
2. Conflict during undo → Warn that undo may fail
3. Conflict timeout (user doesn't respond) → Auto-resolve after 30s
4. Cascading conflicts (resolving one creates another) → Limit to 3 levels

**Test Scenarios:**
- Simulate edit-delete conflict, verify dialog shown
- User resolves conflict, verify operation applied
- Timeout without resolution, verify auto-resolution
- Test all conflict types with UI

---

### FR-10: Performance Optimization
**Priority:** P0 (Blocking)

The system MUST optimize for performance:
- **Protocol Alpha:** Render 1000-node graph at 60 FPS
- **Latency:** Operation broadcast to all clients in <100ms (p95)
- **Memory:** Client-side memory usage <100MB for 1000-node graph
- **Bandwidth:** WebSocket traffic <1MB/min per client

**Optimization Strategies:**
1. **Virtual Rendering:** Only render visible nodes (viewport culling)
2. **Lazy Loading:** Fetch node details on-demand
3. **Batch Operations:** Combine multiple ops into single broadcast
4. **Compression:** Use gzip for WebSocket messages
5. **Debouncing:** Throttle cursor position updates (max 10/sec)
6. **Caching:** Cache graph layout (positions) locally
7. **Web Workers:** Run OT algorithm in background thread

**Performance Monitoring:**
```typescript
class PerformanceMonitor {
    private metrics: Map<string, number[]> = new Map()

    measure(metric: string, value: number) {
        if (!this.metrics.has(metric)) {
            this.metrics.set(metric, [])
        }
        this.metrics.get(metric)!.push(value)
    }

    getP95(metric: string): number {
        const values = this.metrics.get(metric) || []
        if (values.length === 0) return 0

        const sorted = values.slice().sort((a, b) => a - b)
        const index = Math.floor(sorted.length * 0.95)
        return sorted[index]
    }

    report(): PerformanceReport {
        return {
            render_fps: this.getP95("render_fps"),
            broadcast_latency_ms: this.getP95("broadcast_latency_ms"),
            memory_usage_mb: performance.memory.usedJSHeapSize / 1024 / 1024,
        }
    }
}
```

**Edge Cases:**
1. Graph with 10,000+ nodes → Pagination, show top 1000 by relevance
2. Rapid operations (100/sec) → Queue and batch
3. Slow client (old device) → Reduce update frequency
4. Network congestion → Adaptive compression

**Test Scenarios:**
- Load test: 100 clients, 1000 ops/sec, verify p95 latency <100ms
- Render test: 1000-node graph at various zoom levels, measure FPS
- Memory test: Load 5000 nodes, verify <100MB client memory
- Bandwidth test: Measure WebSocket traffic over 1 hour

---

## 2. NON-FUNCTIONAL REQUIREMENTS

### NFR-1: Scalability
- Support 100+ concurrent users per graph
- Handle graphs with up to 10,000 nodes and 50,000 edges
- Horizontal scaling (multiple server instances)

### NFR-2: Reliability
- 99.9% uptime for WebSocket endpoint
- Zero data loss (all operations persisted)
- Graceful degradation (offline mode with eventual sync)

### NFR-3: Security
- Authentication via JWT tokens
- Authorization: Read-only vs. edit permissions
- Rate limiting: Max 100 ops/min per user
- Input validation: Sanitize all user data

### NFR-4: Maintainability
- 100% test coverage (enforced by quality gate)
- Full API documentation (OpenAPI/Swagger)
- Comprehensive logging and monitoring

---

## 3. INTEGRATION POINTS

### Backend (FastAPI)
- **Framework:** FastAPI with WebSocket support
- **Database:** ParagonDB for graph storage
- **Caching:** Redis for session state
- **Message Queue:** RabbitMQ for operation buffering

### Frontend (React)
- **Framework:** React 18 with TypeScript
- **State Management:** Zustand
- **Canvas Rendering:** HTML5 Canvas or React Flow
- **WebSocket Client:** Native WebSocket API or Socket.IO

### Infrastructure
- **Deployment:** Docker + Kubernetes
- **CDN:** Cloudflare for static assets
- **Monitoring:** Prometheus + Grafana
- **Logging:** Structured logs (JSON) to ElasticSearch

---

## 4. TEST SCENARIOS

### Unit Tests
1. OT transform function (all operation pairs)
2. Cycle detection algorithm
3. Vector clock operations
4. Undo/redo stack
5. Constraint validators

### Integration Tests
1. WebSocket connection lifecycle
2. Operation broadcast to multiple clients
3. State synchronization after disconnect
4. Conflict resolution pipeline

### E2E Tests
1. Full workflow: User joins, edits graph, other user sees changes
2. Concurrent editing: 10 users edit simultaneously, all converge
3. Undo cascade: User A creates node, User B adds edge, User A undoes
4. Large graph: Load 1000-node graph, verify performance

### Load Tests
1. 100 concurrent clients
2. 1000 operations/sec throughput
3. 10,000-node graph rendering
4. Network partition recovery

---

## 5. SUCCESS CRITERIA

### Quality Metrics (Hard Constraints)
- **Test Pass Rate:** 100%
- **Static Analysis:** 0 critical issues
- **Graph Invariants:** 100% compliance (DAG, teleology)
- **Code Coverage:** 100%
- **Security:** Pass OWASP Top 10 checklist

### Performance Metrics
- **Render FPS:** 60 FPS for 1000-node graph (p95)
- **Broadcast Latency:** <100ms (p95)
- **Memory Usage:** <100MB client-side
- **Concurrent Users:** 100+ per graph

### Correctness Metrics
- **Convergence:** 100% (all clients reach same state)
- **Causality:** 100% (operations respect causal order)
- **Constraint Compliance:** 100% (no DAG violations)

---

## 6. ORCHESTRATOR GUIDANCE

This problem should exercise the full TDD pipeline with frontend + backend orchestration:

1. **DIALECTIC Phase:**
   - Detect ambiguities: "How should conflicts be resolved automatically?"
   - Generate questions: "Should users see each other's cursors?"

2. **RESEARCH Phase:**
   - Research Operational Transformation algorithms (Jupiter, Wave)
   - Research vector clock implementations
   - Research WebSocket best practices
   - Research collaborative editing UX patterns (Figma, Miro)

3. **PLAN Phase:**
   - Decompose into backend components: WSHandler, OTEngine, ConstraintValidator, etc.
   - Decompose into frontend components: GraphCanvas, Node, Edge, Toolbar, etc.
   - Build dependency graph: WSHandler ← OTEngine ← GraphStore

4. **BUILD Phase:**
   - Generate backend code (FastAPI endpoints, WebSocket handlers)
   - Generate frontend code (React components, Zustand store)
   - Generate schemas (msgspec for Python, TypeScript interfaces)

5. **TEST Phase:**
   - Generate backend tests (pytest with WebSocket mocking)
   - Generate frontend tests (React Testing Library, Vitest)
   - Generate E2E tests (Playwright for browser automation)
   - Run quality gate (100% coverage, 0 critical issues)

6. **VERIFICATION Phase:**
   - Validate teleology (all nodes trace to REQ)
   - Validate graph invariants (DAG, handshaking)
   - Generate documentation (README, API docs, architecture diagram)

**Expected Outputs:**
- 20-25 CODE nodes (backend + frontend components)
- 40-50 TEST_SUITE nodes (comprehensive coverage)
- 2 DOC nodes (backend README, frontend README)
- Full graph with 100% teleological integrity
- Live demo deployment (Docker Compose or Kubernetes)

**Stretch Goals:**
- CRDT implementation (alternative to OT)
- Offline mode with IndexedDB persistence
- Graph analytics (community detection, centrality)
- Export/import (JSON, GraphML formats)
