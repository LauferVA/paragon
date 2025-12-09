"""
PARAGON SCHEMAS - The Grammar of the System

If ontology.py is the Dictionary (defining the words we can use),
schemas.py is the Grammar (defining how we structure sentences).

This module defines the core data structures that flow through the graph:
- NodeMetadata: Governance properties embedded in every node
- NodeData: The payload attached to every graph node
- EdgeData: The payload attached to every graph edge
- Serialization helpers for persistence and IPC

Design Principles:
1. STRICT TYPING: msgspec.Struct with no silent type coercion
2. SEPARATE CONTENT FROM METADATA: Heavy content vs. lightweight queryable metadata
3. KW_ONLY: Enforce keyword arguments to prevent positional mix-ups
4. IMMUTABLE IDS: Node/edge IDs are set once and never change
5. MERKLE PROVENANCE: Content-addressable hashing for integrity

Performance Characteristics:
- msgspec.Struct uses ~3x less memory than dict
- Serialization is ~10x faster than Pydantic
- Attribute access is O(1) during graph traversal
"""
import msgspec
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime, timezone
import uuid
import hashlib

# Try to use blake3 for faster hashing, fall back to sha256
try:
    import blake3
    BLAKE3_AVAILABLE = True
except ImportError:
    BLAKE3_AVAILABLE = False

from core.ontology import (
    NodeType,
    NodeStatus,
    EdgeType,
    ApprovalStatus,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def now_utc() -> str:
    """Fast UTC timestamp as ISO8601 string."""
    return datetime.now(timezone.utc).isoformat()


def generate_id() -> str:
    """Generate a new UUID hex string for node/edge IDs."""
    return uuid.uuid4().hex


def compute_hash(content: str, dependency_hashes: Optional[List[str]] = None) -> str:
    """
    Compute a Merkle hash for content-addressable identification.

    Hash = Hash(Content + Sorted(Dependency_Hashes))

    This creates a chain of provenance - if any upstream content changes,
    all downstream hashes become invalid.

    Args:
        content: The node's content to hash
        dependency_hashes: Hashes of dependencies (sorted for determinism)

    Returns:
        Hex string hash
    """
    # Build the hash input: content + sorted dependency hashes
    hash_input = content
    if dependency_hashes:
        sorted_deps = sorted(dependency_hashes)
        hash_input += "|" + "|".join(sorted_deps)

    data = hash_input.encode('utf-8')

    if BLAKE3_AVAILABLE:
        return blake3.blake3(data).hexdigest()
    else:
        return hashlib.sha256(data).hexdigest()


# =============================================================================
# NODE METADATA (Governance as Data)
# =============================================================================

class NodeMetadata(msgspec.Struct, kw_only=True, frozen=False):
    """
    Governance properties embedded in every node.

    These are PHYSICS, not policy. A node physically cannot be
    processed if it violates these constraints. This moves governance
    from "middleware that can be bypassed" to "data that cannot."

    Uses msgspec.Struct for hot-path performance.
    Accessed during every graph traversal and state transition check.
    """
    # === Cost Governance ===
    cost_limit: Optional[float] = None  # Max USD cost allowed. None = unlimited.
    cost_actual: float = 0.0            # Actual cost incurred.

    # === Security ===
    security_level: int = 0             # 0=public, 1=internal, 2=confidential, 3=restricted
    required_clearance: int = 0         # Minimum clearance level to process

    # === Versioning ===
    version: str = "1.0.0"
    parent_version: Optional[str] = None  # Version of node this was derived from

    # === Approval Workflow ===
    approval_status: ApprovalStatus = "none"
    approved_by: Optional[str] = None   # Agent/user who approved
    approval_required: bool = False     # Whether this node requires approval

    # === Retry Tracking ===
    attempts: int = 0                   # Number of processing attempts so far
    max_attempts: int = 3               # Maximum attempts before escalation
    last_error: Optional[str] = None    # Error message from last failed attempt

    # === Timing ===
    timeout_seconds: Optional[int] = None  # Max processing time. None = no limit.
    deadline: Optional[str] = None      # ISO8601 deadline for completion

    # === Extension Point ===
    extra: Dict[str, Any] = msgspec.field(default_factory=dict)

    def is_cost_exceeded(self) -> bool:
        """Check if cost limit has been exceeded."""
        if self.cost_limit is None:
            return False
        return self.cost_actual >= self.cost_limit

    def is_max_attempts_exceeded(self) -> bool:
        """Check if max retry attempts exceeded."""
        return self.attempts >= self.max_attempts

    def increment_attempt(self) -> "NodeMetadata":
        """Return new metadata with incremented attempt count."""
        return msgspec.structs.replace(self, attempts=self.attempts + 1)


# =============================================================================
# NODE DATA (The Core Graph Payload)
# =============================================================================

class NodeData(msgspec.Struct, kw_only=True, frozen=False):
    """
    The rigid payload attached to every node in the rustworkx graph.

    This is stored directly in rx.PyDiGraph.add_node(). Because it's a
    msgspec.Struct, it uses significantly less memory than a dict and
    allows O(1) attribute access during traversal.

    Architecture Notes:
    - `id`: Business logic UUID (string), NOT the rustworkx integer index
    - `content`: The actual artifact (code, requirement text, etc.)
    - `metadata`: Governance properties from ontology.NodeMetadata

    Memory Strategy:
    - For <100K nodes: Store content directly (current implementation)
    - For >100K nodes: Move content to external store, keep reference here
    """
    # === Identity ===
    id: str                                    # Business UUID (not rx index)
    type: str                                  # NodeType.value (e.g., "CODE")
    status: str = NodeStatus.PENDING.value     # NodeStatus.value

    # === Content ===
    # The actual payload - varies by node type:
    # - REQ: requirement text
    # - CODE: source code
    # - SPEC: specification details
    # - RESEARCH: structured research artifact
    content: str = ""                          # Primary content (text/code)

    # === Structured Data ===
    # Additional structured data beyond the primary content
    # Used for: file paths, function signatures, test results, etc.
    #
    # Message-to-Node Mapping Fields (for dialogue-to-graph correspondence):
    # - dialogue_turn_id: Optional[str] - ID of dialogue turn that defined this node
    # - message_ids: List[str] - IDs of messages that reference this node
    # - definition_turn: Optional[str] - Turn ID when this node was first defined
    # - referenced_in_turns: List[str] - Turn IDs that mention this node
    # - hover_metadata: Dict[str, Any] - Metadata for UI hover display
    #   {
    #     "phase": str - Current phase (research, plan, build, test)
    #     "created_by": str - Agent or human that created this
    #     "created_at": str - ISO timestamp
    #     "status": str - Current node status
    #     "teleology_status": str - Teleology verification status
    #     "related_nodes": List[str] - IDs of related nodes
    #     "key_findings": Optional[List[str]] - Key findings/discoveries
    #   }
    data: Dict[str, Any] = msgspec.field(default_factory=dict)

    # === Governance ===
    metadata: NodeMetadata = msgspec.field(default_factory=NodeMetadata)

    # === Provenance ===
    created_by: str = "system"                 # Agent ID or "human"
    created_at: str = msgspec.field(default_factory=now_utc)
    updated_at: str = msgspec.field(default_factory=now_utc)

    # === Integrity ===
    version: int = 1                           # Incremented on each update
    checksum: Optional[str] = None             # Content hash for verification

    # === Merkle Provenance (Layer 8) ===
    # Hash = Hash(Content + Sorted(Dependency_Hashes))
    # This creates a chain of provenance - if upstream changes, hash invalidates
    merkle_hash: Optional[str] = None

    # === Semantic Embedding (Hybrid Context Assembly) ===
    # 384-dimensional vector from all-MiniLM-L6-v2
    # Pre-computed and stored with node for O(1) similarity lookup
    embedding: Optional[List[float]] = None

    # === Teleology Status (Layer 8) ===
    # Tracks whether this node has a valid chain of causation to a REQ
    # Values: "justified", "unjustified", "root", "orphaned", "unchecked"
    teleology_status: str = "unchecked"

    def touch(self) -> None:
        """Update the updated_at timestamp and increment version."""
        self.updated_at = now_utc()
        self.version += 1

    def set_status(self, new_status: str) -> None:
        """Update status with automatic timestamp refresh."""
        self.status = new_status
        self.touch()

    def add_cost(self, amount: float) -> None:
        """Add to the cost_actual in metadata."""
        self.metadata = msgspec.structs.replace(
            self.metadata,
            cost_actual=self.metadata.cost_actual + amount
        )
        self.touch()

    def increment_attempt(self) -> None:
        """Increment the attempt counter in metadata."""
        self.metadata = msgspec.structs.replace(
            self.metadata,
            attempts=self.metadata.attempts + 1
        )
        self.touch()

    def is_processable(self) -> bool:
        """Check if node can be processed (cost and attempts under limits)."""
        return (
            not self.metadata.is_cost_exceeded() and
            not self.metadata.is_max_attempts_exceeded()
        )

    def compute_merkle_hash(self, dependency_hashes: Optional[List[str]] = None) -> str:
        """
        Compute and store the Merkle hash for this node.

        Args:
            dependency_hashes: Hashes of dependency nodes (from DEPENDS_ON edges)

        Returns:
            The computed hash
        """
        self.merkle_hash = compute_hash(self.content, dependency_hashes)
        self.touch()
        return self.merkle_hash

    def set_teleology_status(self, status: str) -> None:
        """Update teleology status with automatic timestamp refresh."""
        self.teleology_status = status
        self.touch()

    @classmethod
    def create(
        cls,
        type: str,
        content: str = "",
        created_by: str = "system",
        **kwargs
    ) -> "NodeData":
        """Factory method to create a new NodeData with optional custom ID."""
        # Use provided id or generate one
        node_id = kwargs.pop("id", None) or generate_id()
        return cls(
            id=node_id,
            type=type,
            content=content,
            created_by=created_by,
            **kwargs
        )

    @classmethod
    def from_type(cls, node_type: NodeType, content: str = "", **kwargs) -> "NodeData":
        """Create NodeData from a NodeType enum."""
        return cls.create(type=node_type.value, content=content, **kwargs)


# =============================================================================
# EDGE DATA (The Graph Relationship Payload)
# =============================================================================

class EdgeData(msgspec.Struct, kw_only=True, frozen=False):
    """
    The rigid payload attached to every edge in the rustworkx graph.

    Edges are intentionally "thin" - they carry relationship semantics
    and weights, not heavy data. This enables fast graph traversal.

    The source_id and target_id are business UUIDs, not rustworkx indices.
    The ParagonDB handles the translation via its index maps.
    """
    # === Identity ===
    source_id: str                             # Source node UUID
    target_id: str                             # Target node UUID
    type: str                                  # EdgeType.value (e.g., "DEPENDS_ON")

    # === Properties ===
    weight: float = 1.0                        # For pathfinding algorithms

    # === Context ===
    # Optional metadata explaining the relationship
    # e.g., {"reason": "spec_main imports from spec_utils"}
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)

    # === Provenance ===
    created_by: str = "system"
    created_at: str = msgspec.field(default_factory=now_utc)

    @classmethod
    def create(
        cls,
        source_id: str,
        target_id: str,
        type: str,
        **kwargs
    ) -> "EdgeData":
        """Factory method to create an EdgeData."""
        return cls(
            source_id=source_id,
            target_id=target_id,
            type=type,
            **kwargs
        )

    @classmethod
    def depends_on(cls, source_id: str, target_id: str, **kwargs) -> "EdgeData":
        """Create a DEPENDS_ON edge (source depends on target)."""
        return cls.create(
            source_id=source_id,
            target_id=target_id,
            type=EdgeType.DEPENDS_ON.value,
            **kwargs
        )

    @classmethod
    def implements(cls, code_id: str, spec_id: str, **kwargs) -> "EdgeData":
        """Create an IMPLEMENTS edge (code implements spec)."""
        return cls.create(
            source_id=code_id,
            target_id=spec_id,
            type=EdgeType.IMPLEMENTS.value,
            **kwargs
        )


# =============================================================================
# SERIALIZATION HELPERS (The Control Plane Glue)
# =============================================================================

# Pre-compiled encoders/decoders for maximum throughput
# Reuse these instances across the application to avoid recompilation costs

_node_encoder = msgspec.json.Encoder()
_node_decoder = msgspec.json.Decoder(type=NodeData)

_edge_encoder = msgspec.json.Encoder()
_edge_decoder = msgspec.json.Decoder(type=EdgeData)

_node_list_decoder = msgspec.json.Decoder(type=List[NodeData])
_edge_list_decoder = msgspec.json.Decoder(type=List[EdgeData])


def serialize_node(node: NodeData) -> bytes:
    """
    Serialize a NodeData to JSON bytes.

    Uses pre-compiled encoder for maximum performance.
    Zero-copy where possible.
    """
    return _node_encoder.encode(node)


def deserialize_node(data: bytes) -> NodeData:
    """
    Deserialize JSON bytes to a NodeData.

    Uses pre-compiled decoder for maximum performance.
    """
    return _node_decoder.decode(data)


def serialize_edge(edge: EdgeData) -> bytes:
    """Serialize an EdgeData to JSON bytes."""
    return _edge_encoder.encode(edge)


def deserialize_edge(data: bytes) -> EdgeData:
    """Deserialize JSON bytes to an EdgeData."""
    return _edge_decoder.decode(data)


def serialize_nodes(nodes: List[NodeData]) -> bytes:
    """Serialize a list of NodeData to JSON bytes."""
    return _node_encoder.encode(nodes)


def deserialize_nodes(data: bytes) -> List[NodeData]:
    """Deserialize JSON bytes to a list of NodeData."""
    return _node_list_decoder.decode(data)


def serialize_edges(edges: List[EdgeData]) -> bytes:
    """Serialize a list of EdgeData to JSON bytes."""
    return _edge_encoder.encode(edges)


def deserialize_edges(data: bytes) -> List[EdgeData]:
    """Deserialize JSON bytes to a list of EdgeData."""
    return _edge_list_decoder.decode(data)


# =============================================================================
# MSGPACK SERIALIZATION (Binary Format for IPC)
# =============================================================================

_msgpack_encoder = msgspec.msgpack.Encoder()
_msgpack_node_decoder = msgspec.msgpack.Decoder(type=NodeData)
_msgpack_edge_decoder = msgspec.msgpack.Decoder(type=EdgeData)


def serialize_node_msgpack(node: NodeData) -> bytes:
    """Serialize NodeData to msgpack bytes (more compact than JSON)."""
    return _msgpack_encoder.encode(node)


def deserialize_node_msgpack(data: bytes) -> NodeData:
    """Deserialize msgpack bytes to NodeData."""
    return _msgpack_node_decoder.decode(data)


# =============================================================================
# TYPE ALIASES FOR GRAPH OPERATIONS
# =============================================================================

# These aliases make the graph_db.py code more readable
NodePayload = NodeData
EdgePayload = EdgeData

# For batch operations
NodeBatch = List[NodeData]
EdgeBatch = List[EdgeData]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_node_type(type_str: str) -> bool:
    """Check if a string is a valid NodeType value."""
    return type_str in {nt.value for nt in NodeType}


def validate_edge_type(type_str: str) -> bool:
    """Check if a string is a valid EdgeType value."""
    return type_str in {et.value for et in EdgeType}


def validate_status(status_str: str) -> bool:
    """Check if a string is a valid NodeStatus value."""
    return status_str in {ns.value for ns in NodeStatus}
