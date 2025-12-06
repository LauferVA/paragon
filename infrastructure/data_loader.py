"""
PARAGON DATA LOADER - The Digestion System

This module handles data ingestion with two key principles:
1. LAZY EVALUATION: Use Polars LazyFrames to defer computation
2. BATCH OPERATIONS: Cross Python/Rust boundary once, not N times

Architecture:
- PolarsLoader: Lazy file loading with schema validation
- BulkIngestor: Efficient graph population from DataFrames
- Schema definitions for CSV/Parquet import formats

Performance Strategy:
- scan_csv() instead of read_csv() for lazy evaluation
- Filter pushdown to minimize data movement
- Batch insert via add_nodes_from() instead of iterative add_node()
- Column extraction to lists for Rust boundary crossing
"""
import polars as pl
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from core.schemas import NodeData, EdgeData, NodeMetadata
from core.ontology import NodeType, NodeStatus, EdgeType
from core.graph_db import ParagonDB


# =============================================================================
# SCHEMA DEFINITIONS (For Import Validation)
# =============================================================================

# Expected columns for node CSV import
NODE_CSV_SCHEMA = {
    "id": pl.Utf8,
    "type": pl.Utf8,
    "content": pl.Utf8,
    "status": pl.Utf8,
    "created_by": pl.Utf8,
}

# Optional columns that may be present
NODE_CSV_OPTIONAL = {
    "created_at": pl.Utf8,
    "updated_at": pl.Utf8,
    "version": pl.Int64,
    "cost_limit": pl.Float64,
    "cost_actual": pl.Float64,
}

# Expected columns for edge CSV import
EDGE_CSV_SCHEMA = {
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "type": pl.Utf8,
}

# Optional edge columns
EDGE_CSV_OPTIONAL = {
    "weight": pl.Float64,
    "created_by": pl.Utf8,
}


# =============================================================================
# VALIDATION ERRORS
# =============================================================================

class DataLoadError(Exception):
    """Base exception for data loading errors."""
    pass


class SchemaValidationError(DataLoadError):
    """Raised when data doesn't match expected schema."""
    def __init__(self, missing_columns: List[str], invalid_types: Dict[str, str] = None):
        self.missing_columns = missing_columns
        self.invalid_types = invalid_types or {}
        msg = f"Schema validation failed. Missing columns: {missing_columns}"
        if invalid_types:
            msg += f", Invalid types: {invalid_types}"
        super().__init__(msg)


class DataIntegrityError(DataLoadError):
    """Raised when data has integrity issues (null IDs, invalid types, etc.)."""
    pass


# =============================================================================
# POLARS LOADER (Lazy File Loading)
# =============================================================================

class PolarsLoader:
    """
    Lazy data loader using Polars scan operations.

    Key Principle: Never call .collect() until absolutely necessary.
    This allows Polars to optimize the query plan and push down filters.

    Usage:
        loader = PolarsLoader()

        # Lazy load - no data read yet
        lf = loader.load_csv_nodes("nodes.csv")

        # Add filters - still lazy
        lf = lf.filter(pl.col("type") == "CODE")

        # Only now is data read (with filter pushdown)
        df = lf.collect()
    """

    def __init__(self, validate: bool = True):
        """
        Initialize the loader.

        Args:
            validate: If True, validate schema on load. Default True.
        """
        self.validate = validate

    # =========================================================================
    # CSV Loading (Lazy)
    # =========================================================================

    def load_csv_nodes(
        self,
        path: str | Path,
        has_header: bool = True,
        separator: str = ",",
    ) -> pl.LazyFrame:
        """
        Lazy load nodes from CSV file.

        Uses scan_csv() for lazy evaluation with filter pushdown.
        Data is not read until .collect() is called.

        Args:
            path: Path to CSV file
            has_header: Whether CSV has header row
            separator: Column separator character

        Returns:
            LazyFrame that can be filtered before collection

        Raises:
            SchemaValidationError: If validation enabled and schema invalid
        """
        path = Path(path)

        # Lazy scan - no data read yet
        lf = pl.scan_csv(
            path,
            has_header=has_header,
            separator=separator,
            infer_schema_length=1000,
        )

        if self.validate:
            self._validate_node_schema(lf)

        return lf

    def load_csv_edges(
        self,
        path: str | Path,
        has_header: bool = True,
        separator: str = ",",
    ) -> pl.LazyFrame:
        """
        Lazy load edges from CSV file.

        Args:
            path: Path to CSV file
            has_header: Whether CSV has header row
            separator: Column separator character

        Returns:
            LazyFrame for edges
        """
        path = Path(path)

        lf = pl.scan_csv(
            path,
            has_header=has_header,
            separator=separator,
            infer_schema_length=1000,
        )

        if self.validate:
            self._validate_edge_schema(lf)

        return lf

    # =========================================================================
    # Parquet Loading (Lazy)
    # =========================================================================

    def load_parquet_nodes(self, path: str | Path) -> pl.LazyFrame:
        """
        Lazy load nodes from Parquet file.

        Parquet is more efficient than CSV for large datasets:
        - Columnar storage allows reading only needed columns
        - Built-in compression
        - Schema embedded in file

        Args:
            path: Path to Parquet file

        Returns:
            LazyFrame for nodes
        """
        path = Path(path)
        lf = pl.scan_parquet(path)

        if self.validate:
            self._validate_node_schema(lf)

        return lf

    def load_parquet_edges(self, path: str | Path) -> pl.LazyFrame:
        """Lazy load edges from Parquet file."""
        path = Path(path)
        lf = pl.scan_parquet(path)

        if self.validate:
            self._validate_edge_schema(lf)

        return lf

    # =========================================================================
    # Arrow IPC Loading (For Inter-Process Communication)
    # =========================================================================

    def load_arrow_nodes(self, path: str | Path) -> pl.LazyFrame:
        """
        Lazy load nodes from Arrow IPC file.

        Arrow IPC is faster than Parquet for IPC scenarios:
        - No decompression overhead
        - Zero-copy reads possible
        - Ideal for Cosmograph visualization pipeline
        """
        path = Path(path)
        lf = pl.scan_ipc(path)

        if self.validate:
            self._validate_node_schema(lf)

        return lf

    def load_arrow_edges(self, path: str | Path) -> pl.LazyFrame:
        """Lazy load edges from Arrow IPC file."""
        path = Path(path)
        lf = pl.scan_ipc(path)

        if self.validate:
            self._validate_edge_schema(lf)

        return lf

    # =========================================================================
    # Schema Validation
    # =========================================================================

    def _validate_node_schema(self, lf: pl.LazyFrame) -> None:
        """
        Validate that LazyFrame has required node columns AND correct types.

        Note: Uses collect_schema() to get schema without full data collection.
        """
        schema = lf.collect_schema()
        missing = []
        invalid_types = {}

        for col_name, expected_type in NODE_CSV_SCHEMA.items():
            if col_name not in schema:
                missing.append(col_name)
            elif schema[col_name] != expected_type:
                # Track type mismatches for error reporting
                invalid_types[col_name] = f"Expected {expected_type}, got {schema[col_name]}"

        if missing or invalid_types:
            raise SchemaValidationError(missing_columns=missing, invalid_types=invalid_types)

    def _validate_edge_schema(self, lf: pl.LazyFrame) -> None:
        """Validate edge LazyFrame schema AND types."""
        schema = lf.collect_schema()
        missing = []
        invalid_types = {}

        for col_name, expected_type in EDGE_CSV_SCHEMA.items():
            if col_name not in schema:
                missing.append(col_name)
            elif schema[col_name] != expected_type:
                invalid_types[col_name] = f"Expected {expected_type}, got {schema[col_name]}"

        if missing or invalid_types:
            raise SchemaValidationError(missing_columns=missing, invalid_types=invalid_types)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def load_graph_state(
        self,
        nodes_path: str | Path,
        edges_path: str | Path,
        format: str = "parquet",
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        Load both nodes and edges from files.

        Args:
            nodes_path: Path to nodes file
            edges_path: Path to edges file
            format: File format ("csv", "parquet", "arrow")

        Returns:
            Tuple of (nodes_lf, edges_lf)
        """
        loaders = {
            "csv": (self.load_csv_nodes, self.load_csv_edges),
            "parquet": (self.load_parquet_nodes, self.load_parquet_edges),
            "arrow": (self.load_arrow_nodes, self.load_arrow_edges),
        }

        if format not in loaders:
            raise ValueError(f"Unknown format: {format}. Use: {list(loaders.keys())}")

        node_loader, edge_loader = loaders[format]
        return node_loader(nodes_path), edge_loader(edges_path)


# =============================================================================
# BULK INGESTOR (DataFrame -> Graph)
# =============================================================================

class BulkIngestor:
    """
    Efficiently populate ParagonDB from Polars DataFrames.

    Key Optimization: Extract columns as Python lists and use batch insert.
    This crosses the Python/Rust boundary ONCE for N items, not N times.

    Anti-pattern (slow):
        for row in df.iter_rows():
            db.add_node(NodeData(...))  # N Rust calls

    Correct pattern (fast):
        nodes = [NodeData(...) for row in df.iter_rows()]
        db.add_nodes_batch(nodes)  # 1 Rust call

    Usage:
        ingestor = BulkIngestor(db)
        ingestor.ingest_nodes(df)
        ingestor.ingest_edges(df)
    """

    def __init__(self, db: ParagonDB):
        """
        Initialize with target database.

        Args:
            db: ParagonDB instance to populate
        """
        self.db = db

    def ingest_nodes(
        self,
        df: pl.DataFrame,
        default_status: str = NodeStatus.PENDING.value,
    ) -> int:
        """
        Bulk ingest nodes from DataFrame.

        Optimization Strategy:
        1. Push all default/null handling into Polars (Rust side)
        2. Use to_dicts() for efficient row iteration
        3. Leverage msgspec.Struct's fast instantiation via **kwargs

        Args:
            df: DataFrame with node data (must have 'id', 'type' columns)
            default_status: Default status for nodes without 'status' column

        Returns:
            Number of nodes ingested

        Raises:
            DataIntegrityError: If required columns missing or data invalid
        """
        if df.is_empty():
            return 0

        # 1. VALIDATION (Fail fast)
        required = {"id", "type"}
        if not required.issubset(df.columns):
            raise DataIntegrityError(f"Missing required columns: {required - set(df.columns)}")

        # 2. PRE-COMPUTE DEFAULTS IN POLARS (Rust side - much faster than Python loops)
        # Add missing columns with defaults, then fill nulls
        columns_to_add = []
        if "content" not in df.columns:
            columns_to_add.append(pl.lit("").alias("content"))
        if "status" not in df.columns:
            columns_to_add.append(pl.lit(default_status).alias("status"))
        if "created_by" not in df.columns:
            columns_to_add.append(pl.lit("system").alias("created_by"))

        if columns_to_add:
            df = df.with_columns(columns_to_add)

        # Fill any remaining nulls in Polars (avoids Python if/else in loop)
        df_clean = df.with_columns([
            pl.col("content").fill_null(""),
            pl.col("status").fill_null(default_status),
            pl.col("created_by").fill_null("system"),
        ])

        # Check for null IDs after cleaning
        if df_clean["id"].null_count() > 0:
            raise DataIntegrityError("Found rows with null IDs")

        # 3. CONVERT TO DICTS (faster than zipping 5 separate lists)
        rows = df_clean.select(["id", "type", "content", "status", "created_by"]).to_dicts()

        # 4. BATCH CREATION via **kwargs (msgspec.Struct is very fast here)
        nodes = [NodeData(**row) for row in rows]

        # 5. SINGLE RUST CALL
        self.db.add_nodes_batch(nodes)

        return len(nodes)

    def ingest_edges(self, df: pl.DataFrame) -> int:
        """
        Bulk ingest edges from DataFrame.

        Optimization Strategy (same as ingest_nodes):
        1. Push defaults into Polars
        2. Use to_dicts() for efficient iteration
        3. Leverage msgspec.Struct's fast **kwargs instantiation

        Args:
            df: DataFrame with edge data (must have 'source_id', 'target_id', 'type')

        Returns:
            Number of edges ingested

        Raises:
            DataIntegrityError: If required columns missing
        """
        if df.is_empty():
            return 0

        # 1. VALIDATION
        required = {"source_id", "target_id", "type"}
        if not required.issubset(df.columns):
            raise DataIntegrityError(f"Missing required columns: {required - set(df.columns)}")

        # 2. PRE-COMPUTE DEFAULTS IN POLARS
        columns_to_add = []
        if "weight" not in df.columns:
            columns_to_add.append(pl.lit(1.0).alias("weight"))
        if "created_by" not in df.columns:
            columns_to_add.append(pl.lit("system").alias("created_by"))

        if columns_to_add:
            df = df.with_columns(columns_to_add)

        # Fill nulls in Polars
        df_clean = df.with_columns([
            pl.col("weight").fill_null(1.0),
            pl.col("created_by").fill_null("system"),
        ])

        # 3. CONVERT TO DICTS
        rows = df_clean.select(["source_id", "target_id", "type", "weight", "created_by"]).to_dicts()

        # 4. BATCH CREATION
        edges = [EdgeData(**row) for row in rows]

        # 5. SINGLE RUST CALL
        self.db.add_edges_batch(edges)

        return len(edges)

    def ingest_from_files(
        self,
        nodes_path: str | Path,
        edges_path: Optional[str | Path] = None,
        format: str = "parquet",
    ) -> Tuple[int, int]:
        """
        Convenience method to load and ingest from files.

        Args:
            nodes_path: Path to nodes file
            edges_path: Optional path to edges file
            format: File format ("csv", "parquet", "arrow")

        Returns:
            Tuple of (nodes_ingested, edges_ingested)
        """
        loader = PolarsLoader(validate=True)

        # Load nodes
        if format == "csv":
            nodes_lf = loader.load_csv_nodes(nodes_path)
        elif format == "parquet":
            nodes_lf = loader.load_parquet_nodes(nodes_path)
        elif format == "arrow":
            nodes_lf = loader.load_arrow_nodes(nodes_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Collect and ingest nodes
        nodes_df = nodes_lf.collect()
        nodes_count = self.ingest_nodes(nodes_df)

        # Load and ingest edges if provided
        edges_count = 0
        if edges_path:
            if format == "csv":
                edges_lf = loader.load_csv_edges(edges_path)
            elif format == "parquet":
                edges_lf = loader.load_parquet_edges(edges_path)
            elif format == "arrow":
                edges_lf = loader.load_arrow_edges(edges_path)

            edges_df = edges_lf.collect()
            edges_count = self.ingest_edges(edges_df)

        return nodes_count, edges_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_graph_from_parquet(
    nodes_path: str | Path,
    edges_path: Optional[str | Path] = None,
) -> ParagonDB:
    """
    Create a ParagonDB populated from Parquet files.

    Args:
        nodes_path: Path to nodes.parquet
        edges_path: Optional path to edges.parquet

    Returns:
        Populated ParagonDB instance
    """
    db = ParagonDB()
    ingestor = BulkIngestor(db)
    ingestor.ingest_from_files(nodes_path, edges_path, format="parquet")
    return db


def load_graph_from_csv(
    nodes_path: str | Path,
    edges_path: Optional[str | Path] = None,
) -> ParagonDB:
    """
    Create a ParagonDB populated from CSV files.

    Args:
        nodes_path: Path to nodes.csv
        edges_path: Optional path to edges.csv

    Returns:
        Populated ParagonDB instance
    """
    db = ParagonDB()
    ingestor = BulkIngestor(db)
    ingestor.ingest_from_files(nodes_path, edges_path, format="csv")
    return db


def create_sample_node_csv(path: str | Path, num_nodes: int = 100) -> None:
    """
    Create a sample nodes CSV file for testing.

    Args:
        path: Output path
        num_nodes: Number of nodes to generate
    """
    import uuid

    df = pl.DataFrame({
        "id": [uuid.uuid4().hex for _ in range(num_nodes)],
        "type": [NodeType.SPEC.value] * num_nodes,
        "content": [f"Sample content {i}" for i in range(num_nodes)],
        "status": [NodeStatus.PENDING.value] * num_nodes,
        "created_by": ["system"] * num_nodes,
    })

    df.write_csv(path)


def create_sample_edge_csv(
    path: str | Path,
    node_ids: List[str],
    edge_probability: float = 0.1,
) -> None:
    """
    Create a sample edges CSV file for testing.

    Creates random edges between provided node IDs.

    Args:
        path: Output path
        node_ids: List of node IDs to connect
        edge_probability: Probability of edge between any two nodes
    """
    import random

    edges = []
    for i, src in enumerate(node_ids):
        for j, tgt in enumerate(node_ids):
            if i < j and random.random() < edge_probability:
                edges.append({
                    "source_id": src,
                    "target_id": tgt,
                    "type": EdgeType.DEPENDS_ON.value,
                    "weight": 1.0,
                })

    if edges:
        df = pl.DataFrame(edges)
        df.write_csv(path)
