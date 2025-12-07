"""
PARAGON FORGE - ADVERSARY MODULE (The Entropy Generator)

Agent D: The Adversary - Controlled Corruption for Testing Recovery Algorithms

This module introduces controlled errors into graph data while maintaining
complete provenance of every modification. The Manifest is the Answer Key
that enables grading any recovery algorithm.

Design Philosophy:
- REPRODUCIBLE: All corruption is seeded for determinism
- TRACEABLE: Every single modification is logged with before/after state
- VARIED: 8 different error types covering common real-world failure modes
- REALISTIC: Error rates and patterns match observed data quality issues

Use Cases:
- Testing graph repair algorithms
- Validating invariant checking
- Simulating data degradation over time
- Benchmarking recovery accuracy

Architecture:
- EntropyModule: Main corruption engine
- Manifest: Complete audit trail of all modifications
- Modification: Single change record
- AdversaryConfig: Configuration for each error type
"""

import msgspec
import random
import string
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timezone
from copy import deepcopy

import rustworkx as rx

from core.schemas import NodeData, EdgeData
from core.graph_db import ParagonDB


# =============================================================================
# SCHEMAS (msgspec.Struct - NOT Pydantic)
# =============================================================================

class AdversaryConfig(msgspec.Struct, kw_only=True, frozen=True):
    """
    Configuration for a specific error type.

    Each error type has different parameters:
    - drop_edges: No params needed
    - drop_nodes: No params needed
    - mutate_strings: property (default "content"), mutation_type
    - mutate_numbers: property, noise_factor
    - swap_labels: No params needed
    - lag_timestamps: lag_seconds
    - duplicate_nodes: No params needed
    - null_properties: properties (list of property names)
    """
    error_type: str                          # One of 8 error types
    rate: float                              # 0.0 to 1.0 probability
    params: Dict[str, Any] = msgspec.field(default_factory=dict)
    seed: Optional[int] = None               # Per-error-type seed (overrides global)


class Modification(msgspec.Struct, kw_only=True):
    """
    Record of a single modification to the graph.

    This is the atomic unit of the Answer Key - every change is logged.
    """
    error_type: str                          # Type of corruption applied
    target_type: str                         # "node" or "edge"
    target_id: str                           # UUID of affected node/edge
    original_value: Any                      # Pre-corruption state
    corrupted_value: Any                     # Post-corruption state
    timestamp: str                           # ISO8601 UTC timestamp
    metadata: Dict[str, Any] = msgspec.field(default_factory=dict)  # Extra context


class Manifest(msgspec.Struct, kw_only=True):
    """
    The Answer Key - complete audit trail of all corruptions.

    This enables:
    - Grading recovery algorithms (compare against original_value)
    - Analyzing error patterns
    - Debugging corruption logic
    - Reproducible test cases
    """
    world_id: str                            # Identifier for this corrupted world
    seed: int                                # Global RNG seed for reproducibility
    total_modifications: int = 0             # Count of changes
    modifications: List[Modification] = msgspec.field(default_factory=list)
    error_summary: Dict[str, int] = msgspec.field(default_factory=dict)  # error_type -> count
    created_at: str = msgspec.field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self, filepath: str) -> None:
        """Save manifest to JSON file."""
        import json
        with open(filepath, 'w') as f:
            # Convert msgspec.Struct to dict for JSON serialization
            encoder = msgspec.json.Encoder()
            data = encoder.encode(self)
            # Decode bytes to str and parse as JSON for pretty printing
            json_data = msgspec.json.decode(data)
            json.dump(json_data, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> "Manifest":
        """Load manifest from JSON file."""
        with open(filepath, 'r') as f:
            data = f.read()
        decoder = msgspec.json.Decoder(type=Manifest)
        return decoder.decode(data.encode())


# =============================================================================
# ENTROPY MODULE (The Corruption Engine)
# =============================================================================

class EntropyModule:
    """
    Controlled corruption engine for graph data.

    Usage:
        adversary = EntropyModule(seed=42)
        adversary.add_error(AdversaryConfig(
            error_type="drop_edges",
            rate=0.05
        ))
        adversary.add_error(AdversaryConfig(
            error_type="mutate_strings",
            rate=0.02,
            params={"property": "content", "mutation_type": "typo"}
        ))

        corrupted_graph = adversary.corrupt(original_graph)
        manifest = adversary.get_manifest()
        manifest.to_json("answer_key.json")
    """

    def __init__(self, seed: Optional[int] = None, world_id: Optional[str] = None):
        """
        Initialize the entropy module.

        Args:
            seed: Global RNG seed for reproducibility (None = random)
            world_id: Identifier for this corruption scenario
        """
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self.world_id = world_id or f"world_{self.seed}"
        self.rng = random.Random(self.seed)

        # Configuration
        self.error_configs: List[AdversaryConfig] = []

        # Tracking
        self.modifications: List[Modification] = []
        self.error_counts: Dict[str, int] = {}

        # Valid error types
        self.valid_errors = {
            "drop_edges",
            "drop_nodes",
            "mutate_strings",
            "mutate_numbers",
            "swap_labels",
            "lag_timestamps",
            "duplicate_nodes",
            "null_properties",
        }

    def add_error(self, config: AdversaryConfig) -> None:
        """
        Add an error type to apply during corruption.

        Args:
            config: Error configuration

        Raises:
            ValueError: If error_type is invalid or rate is out of bounds
        """
        if config.error_type not in self.valid_errors:
            raise ValueError(f"Invalid error type: {config.error_type}. Must be one of {self.valid_errors}")
        if not 0.0 <= config.rate <= 1.0:
            raise ValueError(f"Error rate must be between 0.0 and 1.0, got {config.rate}")

        self.error_configs.append(config)
        self.error_counts[config.error_type] = 0

    def corrupt(self, graph: ParagonDB) -> ParagonDB:
        """
        Apply all configured corruptions to a graph.

        Creates a deep copy of the graph and applies errors in sequence.

        Args:
            graph: Original ParagonDB

        Returns:
            New ParagonDB with corruptions applied
        """
        # Create a new graph to corrupt
        corrupted = ParagonDB(multigraph=graph._graph.multigraph)

        # Copy all nodes and edges
        nodes = graph.get_all_nodes()
        edges = graph.get_all_edges()

        # Deep copy to avoid mutating originals during corruption
        nodes_copy = [self._deep_copy_node(n) for n in nodes]
        edges_copy = [self._deep_copy_edge(e) for e in edges]

        # Build sets for tracking what gets dropped
        dropped_node_ids: Set[str] = set()
        dropped_edge_keys: Set[Tuple[str, str]] = set()

        # Apply errors in order:
        # 1. Drop nodes (must happen first - affects edges)
        # 2. Drop edges
        # 3. All other mutations

        for config in self.error_configs:
            if config.error_type == "drop_nodes":
                self._apply_drop_nodes(nodes_copy, config, dropped_node_ids)

        for config in self.error_configs:
            if config.error_type == "drop_edges":
                self._apply_drop_edges(edges_copy, config, dropped_node_ids, dropped_edge_keys)

        for config in self.error_configs:
            if config.error_type == "mutate_strings":
                self._apply_mutate_strings(nodes_copy, config, dropped_node_ids)
            elif config.error_type == "mutate_numbers":
                self._apply_mutate_numbers(nodes_copy, config, dropped_node_ids)
            elif config.error_type == "swap_labels":
                self._apply_swap_labels(nodes_copy, config, dropped_node_ids)
            elif config.error_type == "lag_timestamps":
                self._apply_lag_timestamps(nodes_copy, config, dropped_node_ids)
            elif config.error_type == "duplicate_nodes":
                self._apply_duplicate_nodes(nodes_copy, config)
            elif config.error_type == "null_properties":
                self._apply_null_properties(nodes_copy, config, dropped_node_ids)

        # Add surviving nodes and edges to new graph
        for node in nodes_copy:
            if node.id not in dropped_node_ids:
                corrupted.add_node(node, allow_duplicate=True)

        for edge in edges_copy:
            key = (edge.source_id, edge.target_id)
            # Skip edges with dropped endpoints or explicitly dropped edges
            if (key not in dropped_edge_keys and
                edge.source_id not in dropped_node_ids and
                edge.target_id not in dropped_node_ids):
                try:
                    corrupted.add_edge(edge, check_cycle=False)
                except Exception:
                    # Edge might fail to add for other reasons, skip it
                    pass

        return corrupted

    def get_manifest(self) -> Manifest:
        """
        Get the complete Answer Key for all corruptions.

        Returns:
            Manifest with all modifications logged
        """
        return Manifest(
            world_id=self.world_id,
            seed=self.seed,
            total_modifications=len(self.modifications),
            modifications=self.modifications,
            error_summary=dict(self.error_counts),
        )

    # =========================================================================
    # ERROR TYPE IMPLEMENTATIONS
    # =========================================================================

    def _apply_drop_nodes(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Drop random nodes to simulate deletion/failure."""
        rng = random.Random(config.seed) if config.seed else self.rng

        for node in nodes:
            if node.id in dropped:
                continue

            if rng.random() < config.rate:
                dropped.add(node.id)
                self._record_modification(
                    error_type="drop_nodes",
                    target_type="node",
                    target_id=node.id,
                    original_value=self._node_to_dict(node),
                    corrupted_value=None,
                )
                self.error_counts["drop_nodes"] += 1

    def _apply_drop_edges(
        self,
        edges: List[EdgeData],
        config: AdversaryConfig,
        dropped_nodes: Set[str],
        dropped_edges: Set[Tuple[str, str]]
    ) -> None:
        """Drop random edges to simulate packet loss/broken links."""
        rng = random.Random(config.seed) if config.seed else self.rng

        for edge in edges:
            # Skip edges with already-dropped endpoints
            if edge.source_id in dropped_nodes or edge.target_id in dropped_nodes:
                continue

            key = (edge.source_id, edge.target_id)
            if key in dropped_edges:
                continue

            if rng.random() < config.rate:
                dropped_edges.add(key)
                self._record_modification(
                    error_type="drop_edges",
                    target_type="edge",
                    target_id=f"{edge.source_id}->{edge.target_id}",
                    original_value=self._edge_to_dict(edge),
                    corrupted_value=None,
                )
                self.error_counts["drop_edges"] += 1

    def _apply_mutate_strings(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Mutate string properties to simulate OCR/transcription errors."""
        rng = random.Random(config.seed) if config.seed else self.rng
        property_name = config.params.get("property", "content")
        mutation_type = config.params.get("mutation_type", "typo")

        for node in nodes:
            if node.id in dropped:
                continue

            if rng.random() < config.rate:
                # Get the property value
                if property_name == "content":
                    original = node.content
                elif property_name == "type":
                    original = node.type
                elif property_name == "status":
                    original = node.status
                elif property_name in node.data:
                    original = node.data[property_name]
                else:
                    continue  # Property not found

                if not isinstance(original, str) or not original:
                    continue  # Skip non-strings or empty

                # Apply mutation
                if mutation_type == "typo":
                    corrupted = self._introduce_typo(original, rng)
                elif mutation_type == "swap":
                    corrupted = self._swap_characters(original, rng)
                elif mutation_type == "delete":
                    corrupted = self._delete_character(original, rng)
                elif mutation_type == "insert":
                    corrupted = self._insert_character(original, rng)
                else:
                    corrupted = self._introduce_typo(original, rng)

                # Apply the mutation
                if property_name == "content":
                    node.content = corrupted
                elif property_name == "type":
                    node.type = corrupted
                elif property_name == "status":
                    node.status = corrupted
                else:
                    node.data[property_name] = corrupted

                self._record_modification(
                    error_type="mutate_strings",
                    target_type="node",
                    target_id=node.id,
                    original_value=original,
                    corrupted_value=corrupted,
                    metadata={"property": property_name, "mutation_type": mutation_type}
                )
                self.error_counts["mutate_strings"] += 1

    def _apply_mutate_numbers(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Add noise to numeric values."""
        rng = random.Random(config.seed) if config.seed else self.rng
        property_name = config.params.get("property", "version")
        noise_factor = config.params.get("noise_factor", 0.1)

        for node in nodes:
            if node.id in dropped:
                continue

            if rng.random() < config.rate:
                # Get numeric property
                if property_name == "version":
                    original = node.version
                elif property_name in node.data:
                    original = node.data[property_name]
                else:
                    continue

                if not isinstance(original, (int, float)):
                    continue

                # Add noise
                noise = rng.uniform(-noise_factor, noise_factor) * original
                corrupted = original + noise

                # Preserve type
                if isinstance(original, int):
                    corrupted = int(round(corrupted))

                # Apply mutation
                if property_name == "version":
                    node.version = corrupted
                else:
                    node.data[property_name] = corrupted

                self._record_modification(
                    error_type="mutate_numbers",
                    target_type="node",
                    target_id=node.id,
                    original_value=original,
                    corrupted_value=corrupted,
                    metadata={"property": property_name, "noise_factor": noise_factor}
                )
                self.error_counts["mutate_numbers"] += 1

    def _apply_swap_labels(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Swap labels/types between random node pairs."""
        rng = random.Random(config.seed) if config.seed else self.rng

        # Get all non-dropped nodes
        available = [n for n in nodes if n.id not in dropped]

        # Number of swaps to perform
        num_swaps = int(len(available) * config.rate / 2)  # Divide by 2 since each swap affects 2 nodes

        for _ in range(num_swaps):
            if len(available) < 2:
                break

            # Pick two random nodes
            idx1, idx2 = rng.sample(range(len(available)), 2)
            node1 = available[idx1]
            node2 = available[idx2]

            # Swap their types
            original_type1 = node1.type
            original_type2 = node2.type

            node1.type = original_type2
            node2.type = original_type1

            self._record_modification(
                error_type="swap_labels",
                target_type="node",
                target_id=node1.id,
                original_value=original_type1,
                corrupted_value=original_type2,
                metadata={"swapped_with": node2.id}
            )

            self._record_modification(
                error_type="swap_labels",
                target_type="node",
                target_id=node2.id,
                original_value=original_type2,
                corrupted_value=original_type1,
                metadata={"swapped_with": node1.id}
            )

            self.error_counts["swap_labels"] = self.error_counts.get("swap_labels", 0) + 2

    def _apply_lag_timestamps(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Shift timestamps to simulate out-of-order delivery."""
        rng = random.Random(config.seed) if config.seed else self.rng
        lag_seconds = config.params.get("lag_seconds", 3600)  # Default 1 hour

        for node in nodes:
            if node.id in dropped:
                continue

            if rng.random() < config.rate:
                # Parse timestamp
                try:
                    from datetime import datetime, timedelta
                    original_ts = node.created_at
                    dt = datetime.fromisoformat(original_ts.replace('Z', '+00:00'))

                    # Apply random lag (can be positive or negative)
                    lag = rng.uniform(-lag_seconds, lag_seconds)
                    dt_lagged = dt + timedelta(seconds=lag)

                    corrupted_ts = dt_lagged.isoformat()
                    node.created_at = corrupted_ts

                    self._record_modification(
                        error_type="lag_timestamps",
                        target_type="node",
                        target_id=node.id,
                        original_value=original_ts,
                        corrupted_value=corrupted_ts,
                        metadata={"lag_seconds": lag}
                    )
                    self.error_counts["lag_timestamps"] += 1
                except Exception:
                    # Skip if timestamp parsing fails
                    continue

    def _apply_duplicate_nodes(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig
    ) -> None:
        """Create duplicate nodes to simulate data entry errors."""
        rng = random.Random(config.seed) if config.seed else self.rng

        duplicates_to_add = []

        for node in nodes:
            if rng.random() < config.rate:
                # Create a duplicate with new ID
                duplicate = self._deep_copy_node(node)
                from core.schemas import generate_id
                new_id = generate_id()
                duplicate.id = new_id

                duplicates_to_add.append(duplicate)

                self._record_modification(
                    error_type="duplicate_nodes",
                    target_type="node",
                    target_id=new_id,
                    original_value=None,
                    corrupted_value=self._node_to_dict(duplicate),
                    metadata={"duplicate_of": node.id}
                )
                self.error_counts["duplicate_nodes"] += 1

        # Add duplicates to the list
        nodes.extend(duplicates_to_add)

    def _apply_null_properties(
        self,
        nodes: List[NodeData],
        config: AdversaryConfig,
        dropped: Set[str]
    ) -> None:
        """Set properties to None/empty to simulate missing data."""
        rng = random.Random(config.seed) if config.seed else self.rng
        properties = config.params.get("properties", ["content"])

        for node in nodes:
            if node.id in dropped:
                continue

            if rng.random() < config.rate:
                for prop in properties:
                    if prop == "content":
                        original = node.content
                        node.content = ""
                        corrupted = ""
                    elif prop == "status":
                        original = node.status
                        node.status = ""
                        corrupted = ""
                    elif prop in node.data:
                        original = node.data[prop]
                        node.data[prop] = None
                        corrupted = None
                    else:
                        continue

                    self._record_modification(
                        error_type="null_properties",
                        target_type="node",
                        target_id=node.id,
                        original_value=original,
                        corrupted_value=corrupted,
                        metadata={"property": prop}
                    )
                    self.error_counts["null_properties"] += 1

    # =========================================================================
    # STRING MUTATION HELPERS
    # =========================================================================

    def _introduce_typo(self, text: str, rng: random.Random) -> str:
        """Introduce a random typo (substitute one character)."""
        if len(text) < 1:
            return text

        idx = rng.randint(0, len(text) - 1)
        chars = list(text)
        # Replace with random letter/digit
        chars[idx] = rng.choice(string.ascii_letters + string.digits)
        return ''.join(chars)

    def _swap_characters(self, text: str, rng: random.Random) -> str:
        """Swap two adjacent characters."""
        if len(text) < 2:
            return text

        idx = rng.randint(0, len(text) - 2)
        chars = list(text)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return ''.join(chars)

    def _delete_character(self, text: str, rng: random.Random) -> str:
        """Delete a random character."""
        if len(text) < 1:
            return text

        idx = rng.randint(0, len(text) - 1)
        return text[:idx] + text[idx + 1:]

    def _insert_character(self, text: str, rng: random.Random) -> str:
        """Insert a random character."""
        idx = rng.randint(0, len(text))
        char = rng.choice(string.ascii_letters + string.digits)
        return text[:idx] + char + text[idx:]

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _deep_copy_node(self, node: NodeData) -> NodeData:
        """Deep copy a node to avoid mutating the original."""
        # msgspec.Struct doesn't support deepcopy directly, so recreate
        return NodeData(
            id=node.id,
            type=node.type,
            status=node.status,
            content=node.content,
            data=deepcopy(node.data),
            metadata=deepcopy(node.metadata),
            created_by=node.created_by,
            created_at=node.created_at,
            updated_at=node.updated_at,
            version=node.version,
            checksum=node.checksum,
            merkle_hash=node.merkle_hash,
            embedding=node.embedding.copy() if node.embedding else None,
            teleology_status=node.teleology_status,
        )

    def _deep_copy_edge(self, edge: EdgeData) -> EdgeData:
        """Deep copy an edge."""
        return EdgeData(
            source_id=edge.source_id,
            target_id=edge.target_id,
            type=edge.type,
            weight=edge.weight,
            metadata=deepcopy(edge.metadata),
            created_by=edge.created_by,
            created_at=edge.created_at,
        )

    def _node_to_dict(self, node: NodeData) -> Dict[str, Any]:
        """Convert node to dict for logging."""
        return {
            "id": node.id,
            "type": node.type,
            "status": node.status,
            "content": node.content[:100] if len(node.content) > 100 else node.content,  # Truncate
            "created_by": node.created_by,
            "created_at": node.created_at,
        }

    def _edge_to_dict(self, edge: EdgeData) -> Dict[str, Any]:
        """Convert edge to dict for logging."""
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type,
            "weight": edge.weight,
        }

    def _record_modification(
        self,
        error_type: str,
        target_type: str,
        target_id: str,
        original_value: Any,
        corrupted_value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a modification to the manifest."""
        mod = Modification(
            error_type=error_type,
            target_type=target_type,
            target_id=target_id,
            original_value=original_value,
            corrupted_value=corrupted_value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )
        self.modifications.append(mod)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_adversary(
    seed: Optional[int] = None,
    preset: Optional[str] = None
) -> EntropyModule:
    """
    Create an EntropyModule with optional preset configurations.

    Presets:
    - "light": Low error rates for subtle corruption
    - "medium": Moderate error rates
    - "heavy": High error rates for stress testing
    - "realistic": Realistic error rates based on observed data quality

    Args:
        seed: RNG seed for reproducibility
        preset: Preset configuration name

    Returns:
        Configured EntropyModule
    """
    adversary = EntropyModule(seed=seed)

    if preset == "light":
        adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.01))
        adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.005,
                                           params={"property": "content", "mutation_type": "typo"}))

    elif preset == "medium":
        adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.05))
        adversary.add_error(AdversaryConfig(error_type="drop_nodes", rate=0.02))
        adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.02,
                                           params={"property": "content", "mutation_type": "typo"}))
        adversary.add_error(AdversaryConfig(error_type="lag_timestamps", rate=0.01,
                                           params={"lag_seconds": 3600}))

    elif preset == "heavy":
        adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.10))
        adversary.add_error(AdversaryConfig(error_type="drop_nodes", rate=0.05))
        adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.05,
                                           params={"property": "content", "mutation_type": "typo"}))
        adversary.add_error(AdversaryConfig(error_type="swap_labels", rate=0.03))
        adversary.add_error(AdversaryConfig(error_type="duplicate_nodes", rate=0.02))
        adversary.add_error(AdversaryConfig(error_type="null_properties", rate=0.02,
                                           params={"properties": ["content"]}))

    elif preset == "realistic":
        # Based on observed data quality issues in real-world systems
        adversary.add_error(AdversaryConfig(error_type="drop_edges", rate=0.03))
        adversary.add_error(AdversaryConfig(error_type="drop_nodes", rate=0.01))
        adversary.add_error(AdversaryConfig(error_type="mutate_strings", rate=0.01,
                                           params={"property": "content", "mutation_type": "typo"}))
        adversary.add_error(AdversaryConfig(error_type="lag_timestamps", rate=0.02,
                                           params={"lag_seconds": 1800}))
        adversary.add_error(AdversaryConfig(error_type="duplicate_nodes", rate=0.005))
        adversary.add_error(AdversaryConfig(error_type="null_properties", rate=0.01,
                                           params={"properties": ["content"]}))

    return adversary
