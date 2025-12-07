"""
PARAGON CONFIG GRAPH - Graph-Native Configuration Management

Wave 3 Refactor: Configuration as Graph Nodes

Instead of reading TOML files throughout the codebase, configuration
is loaded once and stored as CONFIG nodes in the graph. All components
query the graph for their configuration.

Benefits:
1. Single source of truth (the graph)
2. Configuration is queryable via graph traversal
3. Configuration changes are auditable (signature chains)
4. Per-session or per-node config overrides via edges

Usage:
    from infrastructure.config_graph import initialize_config, get_config

    # At startup
    initialize_config(db)

    # To retrieve config
    config = get_config(db, "git")  # Returns dict or None
"""
import msgspec
import json
from typing import Optional, Dict, Any
from pathlib import Path
import warnings

from core.schemas import NodeData, EdgeData, generate_id
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# CONFIG NODE CREATION
# =============================================================================

def load_toml_config() -> Dict[str, Any]:
    """
    Load configuration from paragon.toml.

    Returns:
        Dict with all configuration sections
    """
    try:
        import tomllib
        config_path = Path(__file__).parent.parent / "config" / "paragon.toml"

        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load config from TOML: {e}")
        return {}


def initialize_config(db, config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """
    Initialize CONFIG nodes in the graph from TOML configuration.

    Creates one CONFIG node per top-level section (system, graph, agents, etc.).
    Each node's content is the JSON-serialized section config.

    Args:
        db: ParagonDB instance
        config_dict: Optional pre-loaded config. If None, loads from TOML.

    Returns:
        Dict mapping section name to node ID
    """
    if config_dict is None:
        config_dict = load_toml_config()

    node_ids = {}

    for section_name, section_config in config_dict.items():
        # Create CONFIG node for this section
        node_id = generate_id()
        config_content = json.dumps(section_config, indent=2)

        node = NodeData(
            id=node_id,
            type=NodeType.CONFIG.value,
            status=NodeStatus.VERIFIED.value,
            content=config_content,
            created_by="config_initializer",
        )

        # Add extra metadata
        node.metadata.extra["config_section"] = section_name
        node.metadata.extra["config_version"] = "1.0.0"

        db.add_node(node)
        node_ids[section_name] = node_id

    return node_ids


def get_config(db, section: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve configuration for a section from the graph.

    This is the graph-native way to access configuration:
    instead of reading files, query the graph.

    Args:
        db: ParagonDB instance
        section: Config section name (e.g., "git", "agents", "llm")

    Returns:
        Dict with configuration, or None if not found
    """
    # Find CONFIG nodes
    config_nodes = db.find_nodes(type=NodeType.CONFIG.value)

    for node in config_nodes:
        if node.metadata.extra.get("config_section") == section:
            try:
                return json.loads(node.content)
            except json.JSONDecodeError:
                warnings.warn(f"Failed to parse config for section {section}")
                return None

    return None


def get_all_config(db) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve all configuration from the graph.

    Returns:
        Dict mapping section name to config dict
    """
    result = {}
    config_nodes = db.find_nodes(type=NodeType.CONFIG.value)

    for node in config_nodes:
        section = node.metadata.extra.get("config_section")
        if section:
            try:
                result[section] = json.loads(node.content)
            except json.JSONDecodeError:
                continue

    return result


def update_config(db, section: str, updates: Dict[str, Any]) -> bool:
    """
    Update configuration in the graph.

    Merges updates with existing config and updates the node.

    Args:
        db: ParagonDB instance
        section: Config section name
        updates: Dict of updates to merge

    Returns:
        True if updated, False if section not found
    """
    config_nodes = db.find_nodes(type=NodeType.CONFIG.value)

    for node in config_nodes:
        if node.metadata.extra.get("config_section") == section:
            try:
                current = json.loads(node.content)
                current.update(updates)

                # Update node content
                db.update_node(node.id, content=json.dumps(current, indent=2))
                return True
            except Exception as e:
                warnings.warn(f"Failed to update config for {section}: {e}")
                return False

    return False


# =============================================================================
# AGENT CONFIG HELPERS
# =============================================================================

def create_agent_config(
    db,
    agent_id: str,
    config: Dict[str, Any],
    applies_to_nodes: Optional[list] = None,
) -> str:
    """
    Create an AGENT_CONFIG node for agent-specific configuration.

    Args:
        db: ParagonDB instance
        agent_id: Identifier for the agent
        config: Agent-specific configuration dict
        applies_to_nodes: Optional list of node IDs this config applies to

    Returns:
        The created node ID
    """
    node_id = generate_id()

    node = NodeData(
        id=node_id,
        type=NodeType.AGENT_CONFIG.value,
        status=NodeStatus.VERIFIED.value,
        content=json.dumps(config, indent=2),
        created_by="config_initializer",
    )

    node.metadata.extra["agent_id"] = agent_id

    db.add_node(node)

    # Create APPLIES_TO edges if specified
    if applies_to_nodes:
        for target_id in applies_to_nodes:
            try:
                edge = EdgeData(
                    source_id=node_id,
                    target_id=target_id,
                    type=EdgeType.APPLIES_TO.value,
                )
                db.add_edge(edge)
            except Exception:
                pass  # Target node might not exist

    return node_id


def get_agent_config(db, agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Get configuration for a specific agent.

    Args:
        db: ParagonDB instance
        agent_id: Agent identifier

    Returns:
        Agent config dict, or None if not found
    """
    agent_nodes = db.find_nodes(type=NodeType.AGENT_CONFIG.value)

    for node in agent_nodes:
        if node.metadata.extra.get("agent_id") == agent_id:
            try:
                return json.loads(node.content)
            except json.JSONDecodeError:
                return None

    return None


# =============================================================================
# RESOURCE POLICY HELPERS
# =============================================================================

def create_resource_policy(
    db,
    policy_name: str,
    ram_threshold: float = 90.0,
    cpu_threshold: float = 95.0,
    poll_interval: int = 5,
) -> str:
    """
    Create a RESOURCE_POLICY node.

    Args:
        db: ParagonDB instance
        policy_name: Name for this policy
        ram_threshold: RAM usage threshold (0-100)
        cpu_threshold: CPU usage threshold (0-100)
        poll_interval: Seconds between checks

    Returns:
        The created node ID
    """
    node_id = generate_id()

    policy_config = {
        "ram_threshold_percent": ram_threshold,
        "cpu_threshold_percent": cpu_threshold,
        "poll_interval_seconds": poll_interval,
    }

    node = NodeData(
        id=node_id,
        type=NodeType.RESOURCE_POLICY.value,
        status=NodeStatus.VERIFIED.value,
        content=json.dumps(policy_config, indent=2),
        created_by="config_initializer",
    )

    node.metadata.extra["policy_name"] = policy_name

    db.add_node(node)

    return node_id


def get_resource_policy(db, policy_name: str = "default") -> Optional[Dict[str, Any]]:
    """
    Get resource policy configuration.

    Args:
        db: ParagonDB instance
        policy_name: Policy name (default: "default")

    Returns:
        Policy config dict, or None if not found
    """
    policy_nodes = db.find_nodes(type=NodeType.RESOURCE_POLICY.value)

    for node in policy_nodes:
        if node.metadata.extra.get("policy_name") == policy_name:
            try:
                return json.loads(node.content)
            except json.JSONDecodeError:
                return None

    return None
