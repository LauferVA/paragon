"""
PARAGON.FORGE LINGUIST - Semantic Masking Layer

Agent C: The Linguist
Mission: Transform generic graphs into domain-specific datasets by applying
         thematic overlays that preserve structure while changing semantics.

Architecture:
- Theme-based transformation: Apply domain-specific naming and properties
- Faker integration: Generate realistic synthetic data
- Structure preservation: Graph topology remains intact
- msgspec schemas: Fast, type-safe configuration (NO PYDANTIC)

Design Principles:
1. STRUCTURE IS SACRED: Never modify graph topology, only semantics
2. DETERMINISTIC: Same seed produces same masked graph
3. COMPOSABLE: Themes can be layered or mixed
4. PERFORMANT: O(V+E) transformation with minimal memory overhead
"""

import msgspec
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass
from faker import Faker
import random

# Paragon imports
from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData


# =============================================================================
# SCHEMAS (msgspec.Struct - NO PYDANTIC)
# =============================================================================

class ThemeConfig(msgspec.Struct, kw_only=True):
    """
    User-facing configuration for custom themes.

    Defines patterns and property generators for transforming generic
    graph nodes into domain-specific entities.
    """
    theme_name: str
    node_name_pattern: str  # Format string with {id}, {type}, {index}
    edge_name_pattern: str  # Format string for edge labels
    node_properties: Dict[str, str]  # property_name -> faker_method
    edge_properties: Dict[str, str]  # property_name -> faker_method
    description: str = ""


class Theme(msgspec.Struct, kw_only=True):
    """
    Internal representation of a theme with compiled generators.

    This is the "compiled" form of ThemeConfig, with actual callable
    functions instead of string method names.
    """
    name: str
    description: str
    node_types: List[str]  # Domain-specific node types
    edge_types: List[str]  # Domain-specific edge types
    node_generators: Dict[str, Callable]  # property -> generator function
    edge_generators: Dict[str, Callable]  # property -> generator function
    node_name_pattern: str
    edge_name_pattern: str


# =============================================================================
# BUILT-IN THEME DEFINITIONS
# =============================================================================

class Themes:
    """
    Registry of built-in themes for common domains.

    Each theme provides:
    - Node type mappings (generic -> domain-specific)
    - Edge type mappings
    - Property generators using Faker
    - Naming patterns
    """

    # Theme 1: GENOMICS
    GENOMICS = "genomics"

    # Theme 2: LOGISTICS
    LOGISTICS = "logistics"

    # Theme 3: SOCIAL
    SOCIAL = "social"

    # Theme 4: FINANCE
    FINANCE = "finance"

    # Theme 5: NETWORK
    NETWORK = "network"

    @staticmethod
    def all_themes() -> Set[str]:
        """Return set of all available theme names."""
        return {
            Themes.GENOMICS,
            Themes.LOGISTICS,
            Themes.SOCIAL,
            Themes.FINANCE,
            Themes.NETWORK,
        }


# =============================================================================
# THEME FACTORY
# =============================================================================

class ThemeFactory:
    """
    Builds Theme objects from theme names or ThemeConfig objects.

    Handles the compilation of string-based Faker method names into
    actual callable functions.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the factory with a Faker instance.

        Args:
            seed: Random seed for reproducible data generation
        """
        self.faker = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)

    def build_genomics_theme(self) -> Theme:
        """Build the GENOMICS theme."""
        return Theme(
            name="genomics",
            description="Molecular biology and genetics domain",
            node_types=["Gene", "Protein", "Pathway", "Organism", "Mutation"],
            edge_types=["expresses", "regulates", "interacts_with", "inhibits", "activates"],
            node_generators={
                "sequence": lambda: self._generate_dna_sequence(),
                "organism": lambda: self.faker.random_element([
                    "Homo sapiens", "Mus musculus", "Drosophila melanogaster",
                    "Caenorhabditis elegans", "Saccharomyces cerevisiae"
                ]),
                "chromosome": lambda: f"chr{self.faker.random_int(1, 22)}",
                "gene_symbol": lambda: self._generate_gene_symbol(),
                "protein_family": lambda: self.faker.random_element([
                    "Kinase", "Transcription Factor", "Receptor", "Enzyme", "Structural"
                ]),
                "expression_level": lambda: round(random.uniform(0.1, 100.0), 2),
            },
            edge_generators={
                "confidence_score": lambda: round(random.uniform(0.5, 1.0), 3),
                "interaction_type": lambda: self.faker.random_element([
                    "physical", "genetic", "regulatory", "predicted"
                ]),
                "evidence": lambda: self.faker.random_element([
                    "experimental", "computational", "literature", "database"
                ]),
            },
            node_name_pattern="{type}_{gene_symbol}",
            edge_name_pattern="{edge_type}",
        )

    def build_logistics_theme(self) -> Theme:
        """Build the LOGISTICS theme."""
        return Theme(
            name="logistics",
            description="Supply chain and transportation domain",
            node_types=["Warehouse", "Factory", "Distribution_Center", "Store", "Port"],
            edge_types=["ships_to", "supplies", "receives_from", "transports", "routes_through"],
            node_generators={
                "location": lambda: f"{self.faker.city()}, {self.faker.country()}",
                "capacity": lambda: self.faker.random_int(1000, 100000),
                "inventory_count": lambda: self.faker.random_int(0, 50000),
                "facility_code": lambda: self.faker.bothify("??-####"),
                "operating_hours": lambda: f"{self.faker.random_int(6, 24)} hours/day",
                "manager": lambda: self.faker.name(),
            },
            edge_generators={
                "distance_km": lambda: round(random.uniform(10, 5000), 1),
                "transit_time_hours": lambda: self.faker.random_int(1, 72),
                "shipping_cost": lambda: round(random.uniform(100, 10000), 2),
                "transport_mode": lambda: self.faker.random_element([
                    "truck", "rail", "ship", "air", "pipeline"
                ]),
            },
            node_name_pattern="{type}_{facility_code}",
            edge_name_pattern="{edge_type}",
        )

    def build_social_theme(self) -> Theme:
        """Build the SOCIAL theme."""
        return Theme(
            name="social",
            description="Social network and user interaction domain",
            node_types=["User", "Group", "Post", "Event", "Organization"],
            edge_types=["follows", "friends_with", "likes", "shares", "mentions", "belongs_to"],
            node_generators={
                "username": lambda: self.faker.user_name(),
                "display_name": lambda: self.faker.name(),
                "email": lambda: self.faker.email(),
                "bio": lambda: self.faker.text(max_nb_chars=100),
                "follower_count": lambda: self.faker.random_int(0, 1000000),
                "verified": lambda: self.faker.boolean(chance_of_getting_true=10),
                "join_date": lambda: self.faker.date_between(start_date='-5y', end_date='today').isoformat(),
            },
            edge_generators={
                "timestamp": lambda: self.faker.date_time_between(start_date='-1y', end_date='now').isoformat(),
                "interaction_count": lambda: self.faker.random_int(1, 1000),
                "relationship_type": lambda: self.faker.random_element([
                    "close_friend", "acquaintance", "family", "colleague"
                ]),
            },
            node_name_pattern="{username}",
            edge_name_pattern="{edge_type}",
        )

    def build_finance_theme(self) -> Theme:
        """Build the FINANCE theme."""
        return Theme(
            name="finance",
            description="Financial transactions and account management domain",
            node_types=["Account", "Company", "Transaction", "Portfolio", "Asset"],
            edge_types=["transfers_to", "owns", "invests_in", "borrows_from", "pays"],
            node_generators={
                "account_number": lambda: self.faker.bban(),
                "company_name": lambda: self.faker.company(),
                "balance": lambda: round(random.uniform(0, 1000000), 2),
                "currency": lambda: self.faker.currency_code(),
                "account_type": lambda: self.faker.random_element([
                    "checking", "savings", "investment", "credit", "loan"
                ]),
                "risk_rating": lambda: self.faker.random_element([
                    "low", "medium", "high", "critical"
                ]),
                "created_date": lambda: self.faker.date_between(start_date='-10y', end_date='today').isoformat(),
            },
            edge_generators={
                "amount": lambda: round(random.uniform(10, 100000), 2),
                "transaction_date": lambda: self.faker.date_time_between(start_date='-1y', end_date='now').isoformat(),
                "status": lambda: self.faker.random_element([
                    "pending", "completed", "failed", "reversed"
                ]),
                "transaction_type": lambda: self.faker.random_element([
                    "wire", "ach", "check", "cash", "credit"
                ]),
            },
            node_name_pattern="{type}_{account_number}",
            edge_name_pattern="{edge_type}",
        )

    def build_network_theme(self) -> Theme:
        """Build the NETWORK theme."""
        return Theme(
            name="network",
            description="Computer network infrastructure domain",
            node_types=["Server", "Router", "Switch", "Firewall", "Device", "Database"],
            edge_types=["connects_to", "routes_through", "backs_up_to", "monitors", "secures"],
            node_generators={
                "hostname": lambda: f"{self.faker.word()}-{self.faker.bothify('??##')}",
                "ip_address": lambda: self.faker.ipv4(),
                "mac_address": lambda: self.faker.mac_address(),
                "os": lambda: self.faker.random_element([
                    "Linux", "Windows Server", "FreeBSD", "Ubuntu", "CentOS"
                ]),
                "cpu_cores": lambda: self.faker.random_element([2, 4, 8, 16, 32, 64]),
                "memory_gb": lambda: self.faker.random_element([8, 16, 32, 64, 128, 256]),
                "uptime_days": lambda: self.faker.random_int(0, 365),
                "location": lambda: self.faker.random_element([
                    "datacenter-1", "datacenter-2", "office-hq", "cloud-us-east", "cloud-eu-west"
                ]),
            },
            edge_generators={
                "bandwidth_mbps": lambda: self.faker.random_element([100, 1000, 10000, 40000]),
                "latency_ms": lambda: round(random.uniform(0.1, 100), 2),
                "packet_loss": lambda: round(random.uniform(0, 5), 3),
                "protocol": lambda: self.faker.random_element([
                    "TCP", "UDP", "ICMP", "HTTP", "HTTPS", "SSH"
                ]),
            },
            node_name_pattern="{hostname}",
            edge_name_pattern="{edge_type}",
        )

    def build_from_config(self, config: ThemeConfig) -> Theme:
        """
        Build a Theme from a user-provided ThemeConfig.

        Compiles string-based Faker method names into callable functions.

        Args:
            config: User-provided theme configuration

        Returns:
            Compiled Theme object
        """
        # Compile node generators
        node_generators = {}
        for prop_name, faker_method in config.node_properties.items():
            node_generators[prop_name] = self._compile_faker_method(faker_method)

        # Compile edge generators
        edge_generators = {}
        for prop_name, faker_method in config.edge_properties.items():
            edge_generators[prop_name] = self._compile_faker_method(faker_method)

        return Theme(
            name=config.theme_name,
            description=config.description,
            node_types=["Generic"],  # Custom themes use generic types
            edge_types=["generic_edge"],
            node_generators=node_generators,
            edge_generators=edge_generators,
            node_name_pattern=config.node_name_pattern,
            edge_name_pattern=config.edge_name_pattern,
        )

    def build(self, theme_name: str) -> Theme:
        """
        Build a theme by name.

        Args:
            theme_name: One of the Themes constants

        Returns:
            Compiled Theme object

        Raises:
            ValueError: If theme_name is not recognized
        """
        builders = {
            Themes.GENOMICS: self.build_genomics_theme,
            Themes.LOGISTICS: self.build_logistics_theme,
            Themes.SOCIAL: self.build_social_theme,
            Themes.FINANCE: self.build_finance_theme,
            Themes.NETWORK: self.build_network_theme,
        }

        if theme_name not in builders:
            raise ValueError(
                f"Unknown theme: {theme_name}. "
                f"Available themes: {', '.join(Themes.all_themes())}"
            )

        return builders[theme_name]()

    def _compile_faker_method(self, method_name: str) -> Callable:
        """
        Convert a Faker method name string to a callable function.

        Args:
            method_name: Name of a Faker method (e.g., "name", "email")

        Returns:
            Callable that invokes the Faker method
        """
        if not hasattr(self.faker, method_name):
            raise ValueError(f"Faker has no method: {method_name}")

        method = getattr(self.faker, method_name)
        return lambda: method()

    def _generate_dna_sequence(self, length: int = 20) -> str:
        """Generate a random DNA sequence."""
        bases = ['A', 'T', 'G', 'C']
        return ''.join(random.choice(bases) for _ in range(length))

    def _generate_gene_symbol(self) -> str:
        """Generate a realistic gene symbol."""
        prefixes = ['TP', 'BRCA', 'EGFR', 'KRAS', 'MYC', 'PTEN', 'RB', 'APC']
        prefix = random.choice(prefixes)
        suffix = random.randint(1, 99)
        return f"{prefix}{suffix}" if random.random() > 0.5 else prefix


# =============================================================================
# MASKING LAYER
# =============================================================================

class MaskingLayer:
    """
    The core semantic masking engine.

    Transforms a generic graph into a domain-specific dataset by:
    1. Preserving graph structure (nodes, edges, topology)
    2. Applying thematic node/edge names
    3. Generating realistic properties via Faker
    4. Maintaining deterministic mappings (same seed -> same output)

    Usage:
        masker = MaskingLayer(seed=42)
        themed_graph = masker.apply(graph, theme=Themes.GENOMICS)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the masking layer.

        Args:
            seed: Random seed for reproducible transformations
        """
        self.factory = ThemeFactory(seed=seed)
        self._node_index_map: Dict[str, int] = {}  # node_id -> sequential index

    def apply(
        self,
        graph: ParagonDB,
        theme: str | ThemeConfig,
        preserve_original: bool = False,
    ) -> ParagonDB:
        """
        Apply a theme to a graph, creating a masked copy.

        Args:
            graph: Source graph to transform
            theme: Theme name (string) or ThemeConfig object
            preserve_original: If True, preserve original data in node.data["original"]

        Returns:
            New ParagonDB with theme applied

        Raises:
            ValueError: If theme is invalid
        """
        # Build the theme
        if isinstance(theme, str):
            compiled_theme = self.factory.build(theme)
        elif isinstance(theme, ThemeConfig):
            compiled_theme = self.factory.build_from_config(theme)
        else:
            raise ValueError(f"Invalid theme type: {type(theme)}")

        # Create new graph
        masked_graph = ParagonDB(multigraph=False)

        # Build node index map for deterministic naming
        self._build_node_index_map(graph)

        # Transform nodes
        node_id_map = {}  # old_id -> new_id
        for old_node in graph.iter_nodes():
            new_node = self._transform_node(
                old_node,
                compiled_theme,
                preserve_original=preserve_original
            )
            masked_graph.add_node(new_node)
            node_id_map[old_node.id] = new_node.id

        # Transform edges
        for old_edge in graph.get_all_edges():
            new_edge = self._transform_edge(
                old_edge,
                compiled_theme,
                node_id_map,
                preserve_original=preserve_original
            )
            masked_graph.add_edge(new_edge, check_cycle=False)

        return masked_graph

    def _build_node_index_map(self, graph: ParagonDB) -> None:
        """Build a mapping from node IDs to sequential indices."""
        self._node_index_map = {}
        for idx, node in enumerate(graph.iter_nodes()):
            self._node_index_map[node.id] = idx

    def _transform_node(
        self,
        node: NodeData,
        theme: Theme,
        preserve_original: bool,
    ) -> NodeData:
        """
        Transform a single node according to the theme.

        Args:
            node: Original node
            theme: Compiled theme
            preserve_original: Whether to preserve original data

        Returns:
            New NodeData with themed properties
        """
        # Generate themed properties
        themed_properties = {}
        for prop_name, generator in theme.node_generators.items():
            themed_properties[prop_name] = generator()

        # Assign a domain-specific node type
        node_type_idx = self._node_index_map[node.id] % len(theme.node_types)
        themed_type = theme.node_types[node_type_idx]

        # Generate human-readable label
        label = self._generate_node_label(
            node,
            themed_type,
            themed_properties,
            theme.node_name_pattern
        )

        # Build new node data
        new_data = {
            "label": label,
            "type": themed_type,
            **themed_properties,
        }

        if preserve_original:
            new_data["original"] = {
                "id": node.id,
                "type": node.type,
                "status": node.status,
                "content_preview": node.content[:100] if node.content else "",
            }

        # Create new node with themed content
        return NodeData.create(
            type=themed_type,
            content=f"{label}: {self._generate_themed_content(themed_type, themed_properties)}",
            status=node.status,
            data=new_data,
            created_by=f"linguist_{theme.name}",
        )

    def _transform_edge(
        self,
        edge: EdgeData,
        theme: Theme,
        node_id_map: Dict[str, str],
        preserve_original: bool,
    ) -> EdgeData:
        """
        Transform a single edge according to the theme.

        Args:
            edge: Original edge
            theme: Compiled theme
            node_id_map: Mapping from old node IDs to new node IDs
            preserve_original: Whether to preserve original data

        Returns:
            New EdgeData with themed properties
        """
        # Generate themed properties
        themed_properties = {}
        for prop_name, generator in theme.edge_generators.items():
            themed_properties[prop_name] = generator()

        # Assign a domain-specific edge type
        edge_type_hash = hash(edge.type) % len(theme.edge_types)
        themed_type = theme.edge_types[edge_type_hash]

        # Build metadata
        metadata = themed_properties.copy()
        if preserve_original:
            metadata["original"] = {
                "type": edge.type,
                "weight": edge.weight,
            }

        # Create new edge
        return EdgeData.create(
            source_id=node_id_map[edge.source_id],
            target_id=node_id_map[edge.target_id],
            type=themed_type,
            weight=edge.weight,
            metadata=metadata,
            created_by=f"linguist_{theme.name}",
        )

    def _generate_node_label(
        self,
        node: NodeData,
        themed_type: str,
        properties: Dict[str, Any],
        pattern: str,
    ) -> str:
        """
        Generate a human-readable label for a node.

        Args:
            node: Original node
            themed_type: Themed node type
            properties: Generated properties
            pattern: Label pattern (e.g., "{type}_{id}")

        Returns:
            Formatted label string
        """
        # Build format context
        context = {
            "id": node.id[:8],  # Short ID
            "type": themed_type,
            "index": self._node_index_map[node.id],
            **properties,
        }

        # Try to format the pattern
        try:
            return pattern.format(**context)
        except KeyError as e:
            # Fallback if pattern references missing property
            return f"{themed_type}_{context['index']}"

    def _generate_themed_content(
        self,
        themed_type: str,
        properties: Dict[str, Any],
    ) -> str:
        """
        Generate themed content text for a node.

        Args:
            themed_type: Themed node type
            properties: Generated properties

        Returns:
            Content string summarizing the node
        """
        # Create a readable summary of properties
        prop_strings = []
        for key, value in properties.items():
            if isinstance(value, (int, float, str, bool)):
                prop_strings.append(f"{key}={value}")

        return f"{themed_type} entity with properties: {', '.join(prop_strings)}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def apply_theme(
    graph: ParagonDB,
    theme: str | ThemeConfig,
    seed: Optional[int] = None,
    preserve_original: bool = False,
) -> ParagonDB:
    """
    Convenience function to apply a theme to a graph.

    Args:
        graph: Source graph
        theme: Theme name or ThemeConfig
        seed: Random seed for reproducibility
        preserve_original: Whether to preserve original data

    Returns:
        Themed graph
    """
    masker = MaskingLayer(seed=seed)
    return masker.apply(graph, theme, preserve_original=preserve_original)


def list_available_themes() -> List[str]:
    """Return list of all available built-in themes."""
    return sorted(Themes.all_themes())
