"""
PARAGON WEB RESEARCH TOOLS - Tavily Integration

Tools for web-based research using Tavily search API.

Design:
- Tavily is an industrial-grade, agent-first search provider
- Results are structured and optimized for LLM consumption
- Creates RESEARCH nodes linked to REQ nodes via RESEARCH_FOR edges

Layer 7A Integration:
- Web search provides context for research agents
- Results are stored as structured data in graph nodes
"""
import os
from typing import List, Optional
import msgspec
import logging

from agents.tools import get_db, NodeResult
from core.ontology import NodeType, EdgeType, NodeStatus

logger = logging.getLogger(__name__)

# Try to import Tavily, graceful degradation if unavailable
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("tavily-python not installed. Install with: pip install tavily-python")


# =============================================================================
# SEARCH RESULT SCHEMAS
# =============================================================================

class SearchResult(msgspec.Struct, kw_only=True, frozen=True):
    """
    A single search result from Tavily.

    Contains structured information optimized for agent consumption.
    """
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None


class SearchResponse(msgspec.Struct, kw_only=True, frozen=True):
    """
    Complete response from a web search operation.

    Contains all results and metadata from the search.
    """
    query: str
    results: List[SearchResult]
    total_results: int
    context: str = ""


# =============================================================================
# WEB SEARCH TOOL
# =============================================================================

def search_web(
    query: str,
    context: str = "",
    max_results: int = 5,
    search_depth: str = "advanced",
    create_node: bool = False,
    req_node_id: Optional[str] = None,
) -> SearchResponse:
    """
    Perform web search using Tavily API.

    Args:
        query: The search query
        context: Additional context to refine search (e.g., requirement description)
        max_results: Maximum number of results to return (default: 5)
        search_depth: "basic" or "advanced" (default: "advanced")
        create_node: If True, create a RESEARCH node in the graph
        req_node_id: If provided, link RESEARCH node to this REQ node

    Returns:
        SearchResponse with structured results

    Raises:
        ValueError: If TAVILY_API_KEY environment variable not set
        RuntimeError: If Tavily client not available

    Example:
        >>> results = search_web(
        ...     query="rustworkx graph library best practices",
        ...     context="Building a graph database for code analysis",
        ...     max_results=5
        ... )
        >>> for result in results.results:
        ...     print(f"{result.title}: {result.score}")
    """
    if not TAVILY_AVAILABLE:
        logger.error("Tavily client not available. Install with: pip install tavily-python")
        # Return empty results instead of raising
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            context="Tavily client not available"
        )

    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY environment variable not set")
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            context="TAVILY_API_KEY not configured"
        )

    try:
        # Initialize Tavily client
        client = TavilyClient(api_key=api_key)

        # Perform search with context
        # Tavily's context parameter helps refine results for specific use cases
        search_kwargs = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
        }

        if context:
            # Include context in the query for better relevance
            search_kwargs["query"] = f"{query} {context}"

        raw_response = client.search(**search_kwargs)

        # Parse Tavily response into our structured format
        results = []
        for item in raw_response.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
            )
            results.append(result)

        response = SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            context=context,
        )

        # Optionally create RESEARCH node in graph
        if create_node:
            _create_research_node(
                query=query,
                response=response,
                req_node_id=req_node_id,
            )

        return response

    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            context=f"Search failed: {str(e)}"
        )


# =============================================================================
# GRAPH INTEGRATION
# =============================================================================

def _create_research_node(
    query: str,
    response: SearchResponse,
    req_node_id: Optional[str] = None,
) -> NodeResult:
    """
    Create a RESEARCH node in the graph from search results.

    Args:
        query: The search query
        response: SearchResponse with results
        req_node_id: Optional REQ node to link to

    Returns:
        NodeResult indicating success/failure
    """
    db = get_db()

    # Format search results as content
    content_lines = [f"# Research Query: {query}\n"]

    if response.context:
        content_lines.append(f"**Context:** {response.context}\n")

    content_lines.append(f"\n## Search Results ({response.total_results} found)\n")

    for i, result in enumerate(response.results, 1):
        content_lines.append(f"\n### {i}. {result.title}")
        content_lines.append(f"**URL:** {result.url}")
        content_lines.append(f"**Relevance Score:** {result.score:.2f}")
        if result.published_date:
            content_lines.append(f"**Published:** {result.published_date}")
        content_lines.append(f"\n{result.content}\n")
        content_lines.append("---")

    content = "\n".join(content_lines)

    # Create structured data
    data = {
        "query": query,
        "search_depth": "advanced",
        "total_results": response.total_results,
        "sources": [
            {
                "title": r.title,
                "url": r.url,
                "score": r.score,
            }
            for r in response.results
        ],
    }

    # Create RESEARCH node
    try:
        node = db.add_node(
            node_type=NodeType.RESEARCH.value,
            content=content,
            data=data,
            created_by="researcher_agent",
            status=NodeStatus.PENDING.value,
        )

        # Link to REQ node if provided
        if req_node_id:
            db.add_edge(
                source_id=node.id,
                target_id=req_node_id,
                edge_type=EdgeType.RESEARCH_FOR.value,
                created_by="researcher_agent",
            )

        return NodeResult(
            success=True,
            node_id=node.id,
            message=f"Created RESEARCH node with {response.total_results} results",
        )

    except Exception as e:
        logger.error(f"Failed to create RESEARCH node: {e}")
        return NodeResult(
            success=False,
            node_id="",
            message=f"Failed to create RESEARCH node: {str(e)}",
        )


def create_research_from_results(
    query: str,
    results: List[SearchResult],
    req_node_id: Optional[str] = None,
    context: str = "",
) -> NodeResult:
    """
    Create a RESEARCH node from pre-existing search results.

    Useful when you've already performed a search and want to store
    the results in the graph.

    Args:
        query: The search query
        results: List of SearchResult objects
        req_node_id: Optional REQ node to link to
        context: Additional context about the search

    Returns:
        NodeResult indicating success/failure
    """
    response = SearchResponse(
        query=query,
        results=results,
        total_results=len(results),
        context=context,
    )

    return _create_research_node(
        query=query,
        response=response,
        req_node_id=req_node_id,
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

def check_tavily_config() -> dict:
    """
    Check Tavily configuration and availability.

    Returns:
        Dictionary with configuration status
    """
    return {
        "tavily_available": TAVILY_AVAILABLE,
        "api_key_set": bool(os.getenv("TAVILY_API_KEY")),
        "ready": TAVILY_AVAILABLE and bool(os.getenv("TAVILY_API_KEY")),
    }
