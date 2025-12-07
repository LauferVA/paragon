"""
PARAGON EMBEDDINGS - Semantic Similarity Layer

This module provides the "fuzzy" layer of the hybrid context assembly system.
While the graph edges provide deterministic relationships (compiler logic),
embeddings enable discovery of semantically similar content (fuzzy logic).

Architecture:
- Model: all-MiniLM-L6-v2 (384 dimensions, ~22M params, runs locally)
- Storage: Embeddings pre-computed and stored in NodeData.embedding
- Lookup: O(n) cosine similarity scan, O(1) with approximate nearest neighbors

Design Principles:
1. LAZY INITIALIZATION: Model loaded only when first needed
2. GRACEFUL DEGRADATION: Returns empty results if unavailable
3. BATCH EFFICIENCY: Compute embeddings in batches when possible
4. DETERMINISTIC: Same input always produces same embedding

Use Cases:
- Code generation: Find similar implementations for patterns
- Test generation: Find similar test strategies
- Research: Find related concepts and prior art
- Attribution: Find similar failure patterns
- Quality gate: Find similar quality assessments
- Adaptive questioning: Find similar clarification patterns
"""

from typing import List, Optional, Tuple, Dict, Any
import math

# Lazy loading pattern - model only loaded when needed
_model = None
_model_available = None  # None = unchecked, True/False = result


def _get_model():
    """
    Lazy initialization of the sentence transformer model.

    Uses all-MiniLM-L6-v2:
    - 384 dimensions
    - Fast inference (~4000 sentences/sec on CPU)
    - Good quality for semantic similarity
    - ~22M parameters, runs locally
    """
    global _model, _model_available

    if _model_available is False:
        return None

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
        _model_available = True
        return _model
    except ImportError:
        _model_available = False
        return None
    except Exception:
        _model_available = False
        return None


def is_available() -> bool:
    """Check if embedding functionality is available."""
    global _model_available
    if _model_available is None:
        _get_model()
    return _model_available is True


# =============================================================================
# CORE EMBEDDING FUNCTIONS
# =============================================================================

def compute_embedding(text: str) -> Optional[List[float]]:
    """
    Compute embedding vector for a single text.

    Args:
        text: The text to embed (code, requirement, spec, etc.)

    Returns:
        384-dimensional embedding vector, or None if unavailable
    """
    model = _get_model()
    if model is None:
        return None

    if not text or not text.strip():
        return None

    try:
        # SentenceTransformer returns numpy array
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception:
        return None


def compute_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Compute embeddings for multiple texts efficiently.

    Batching is more efficient than individual calls due to
    GPU/vectorization optimizations in the transformer.

    Args:
        texts: List of texts to embed

    Returns:
        List of embeddings (same order as input), None for failures
    """
    model = _get_model()
    if model is None:
        return [None] * len(texts)

    if not texts:
        return []

    # Filter out empty texts and track indices
    valid_indices = []
    valid_texts = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_indices.append(i)
            valid_texts.append(text)

    if not valid_texts:
        return [None] * len(texts)

    try:
        embeddings = model.encode(valid_texts, convert_to_numpy=True)

        # Reconstruct full result list
        result = [None] * len(texts)
        for i, emb in zip(valid_indices, embeddings):
            result[i] = emb.tolist()

        return result
    except Exception:
        return [None] * len(texts)


# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Returns value in [-1, 1] where:
    - 1.0 = identical direction (semantically identical)
    - 0.0 = orthogonal (unrelated)
    - -1.0 = opposite direction (semantically opposite)

    For text similarity, values typically range [0.3, 1.0].
    Threshold of 0.6+ indicates meaningful semantic similarity.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Cosine similarity score
    """
    if not vec1 or not vec2:
        return 0.0

    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_embeddings(
    query_embedding: List[float],
    candidate_embeddings: List[Tuple[str, List[float]]],
    threshold: float = 0.6,
    limit: int = 5,
) -> List[Tuple[str, float]]:
    """
    Find embeddings most similar to a query.

    This is the core similarity search used by graph_db.find_similar_nodes().

    Args:
        query_embedding: The embedding to search for
        candidate_embeddings: List of (node_id, embedding) tuples
        threshold: Minimum similarity score (0.0-1.0)
        limit: Maximum results to return

    Returns:
        List of (node_id, score) tuples, sorted by score descending
    """
    if not query_embedding or not candidate_embeddings:
        return []

    results = []
    for node_id, embedding in candidate_embeddings:
        if embedding is None:
            continue

        score = cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            results.append((node_id, score))

    # Sort by score descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:limit]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def embedding_dimension() -> int:
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return 384


def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    L2-normalize an embedding vector.

    Normalized vectors enable dot product to equal cosine similarity,
    which is slightly faster for bulk operations.

    Args:
        embedding: Raw embedding vector

    Returns:
        Unit-length embedding vector
    """
    if not embedding:
        return []

    norm = math.sqrt(sum(x * x for x in embedding))
    if norm == 0:
        return embedding

    return [x / norm for x in embedding]


def text_to_embedding_key(text: str) -> str:
    """
    Create a cache key for text embeddings.

    Used for deduplication when the same text appears multiple times.

    Args:
        text: The text to create a key for

    Returns:
        Hash-based key string
    """
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


# =============================================================================
# BATCH UTILITIES FOR GRAPH OPERATIONS
# =============================================================================

def compute_embeddings_for_nodes(
    nodes: List[Dict[str, Any]],
    content_field: str = "content",
) -> Dict[str, List[float]]:
    """
    Compute embeddings for a list of node dictionaries.

    Useful for bulk operations like initial graph load or migration.

    Args:
        nodes: List of node dicts with 'id' and content_field
        content_field: Field name containing text to embed

    Returns:
        Dict mapping node_id to embedding
    """
    if not nodes:
        return {}

    # Extract texts and ids
    node_ids = []
    texts = []
    for node in nodes:
        node_id = node.get('id')
        text = node.get(content_field, '')
        if node_id:
            node_ids.append(node_id)
            texts.append(text)

    # Batch compute
    embeddings = compute_embeddings_batch(texts)

    # Build result dict
    result = {}
    for node_id, embedding in zip(node_ids, embeddings):
        if embedding is not None:
            result[node_id] = embedding

    return result


def update_node_embedding(node_data) -> bool:
    """
    Compute and set embedding on a NodeData object.

    Args:
        node_data: A NodeData instance with 'content' field

    Returns:
        True if embedding was computed, False otherwise
    """
    if not hasattr(node_data, 'content') or not hasattr(node_data, 'embedding'):
        return False

    embedding = compute_embedding(node_data.content)
    if embedding is not None:
        node_data.embedding = embedding
        return True

    return False
