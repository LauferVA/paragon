"""
PARAGON RESEARCH SIMILARITY INTEGRATION - Connect Similarity Search to Research Phase

Integrates the repository scanner into the research phase of the orchestrator.
Runs similarity searches and creates RESEARCH nodes with findings.

Design Principles:
1. ASYNC-AWARE: Compatible with orchestrator's async flow
2. GRAPH-NATIVE: Creates RESEARCH nodes for similarity findings
3. TRANSPARENT: Logs all searches to RerunLogger
4. CONFIGURABLE: Can be enabled/disabled via config
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Try importing similarity components
try:
    from infrastructure.repo_scanner import create_scanner, SimilarProject, SimilarModule, SimilarImplementation
    from agents.similarity_advisor import create_advisor
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    logger.warning("Similarity search components not available")

# Graph integration
try:
    from agents.tools import add_node
    from core.ontology import NodeType, EdgeType
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# =============================================================================
# RESEARCH SIMILARITY ORCHESTRATION
# =============================================================================

async def search_similar_code(
    spec: str,
    session_id: str,
    req_node_id: Optional[str] = None,
    rerun_logger: Optional[Any] = None,
    enable_project_search: bool = True,
    enable_module_search: bool = True,
    enable_implementation_search: bool = False,  # More expensive, off by default
) -> Dict[str, Any]:
    """
    Search for similar code at all three levels.

    Args:
        spec: User's specification/requirement
        session_id: Current session ID
        req_node_id: Optional REQ node to link findings to
        rerun_logger: Optional RerunLogger for transparency
        enable_project_search: Whether to search for similar projects
        enable_module_search: Whether to search for similar modules
        enable_implementation_search: Whether to search for implementations

    Returns:
        Dictionary with similarity findings and formatted suggestions
    """
    if not SIMILARITY_AVAILABLE:
        logger.debug("Similarity search not available")
        return {
            "projects": [],
            "modules": [],
            "implementations": [],
            "suggestions": "Similarity search unavailable (missing dependencies)",
            "research_node_id": None,
        }

    # Create scanner and advisor
    scanner = create_scanner(rerun_logger=rerun_logger)
    advisor = create_advisor()

    # Results
    projects = []
    modules = []
    implementations = []

    try:
        # Level 1: Project-level similarity
        if enable_project_search:
            if rerun_logger:
                rerun_logger.log_thought(
                    "research_similarity",
                    "Searching for similar projects..."
                )

            projects = await scanner.search_similar_projects(
                description=spec,
                sources=["github"],  # Could add "local" when local indexing is ready
                max_results=5,
                min_similarity=0.5,
            )

            logger.info(f"Found {len(projects)} similar projects")

        # Level 2: Module-level similarity
        if enable_module_search:
            if rerun_logger:
                rerun_logger.log_thought(
                    "research_similarity",
                    "Searching for similar modules..."
                )

            modules = await scanner.search_similar_modules(
                module_spec=spec,
                language="python",
                max_results=5,
                min_similarity=0.6,
            )

            logger.info(f"Found {len(modules)} similar modules")

        # Level 3: Implementation-level similarity
        if enable_implementation_search:
            if rerun_logger:
                rerun_logger.log_thought(
                    "research_similarity",
                    "Searching for similar implementations..."
                )

            implementations = await scanner.search_similar_implementations(
                task_description=spec,
                language="python",
                max_results=3,
                min_similarity=0.7,
            )

            logger.info(f"Found {len(implementations)} similar implementations")

        # Format suggestions
        suggestions = advisor.format_summary(
            projects=projects,
            modules=modules,
            implementations=implementations,
        )

        # Create RESEARCH node with findings (if graph available)
        research_node_id = None
        if GRAPH_AVAILABLE and (projects or modules or implementations):
            research_content = advisor.format_research_node_content(
                query=spec,
                projects=projects,
                modules=modules,
                implementations=implementations,
            )

            try:
                result = add_node(
                    node_type=NodeType.RESEARCH.value,
                    content=research_content,
                    data={
                        "research_type": "similarity_search",
                        "project_count": len(projects),
                        "module_count": len(modules),
                        "implementation_count": len(implementations),
                        "session_id": session_id,
                    },
                    created_by="research_similarity_agent",
                )

                if result.success:
                    research_node_id = result.node_id

                    # Link to REQ node if provided
                    if req_node_id:
                        from agents.tools import add_edge
                        add_edge(
                            source_id=research_node_id,
                            target_id=req_node_id,
                            edge_type=EdgeType.RESEARCH_FOR.value,
                        )

                    logger.info(f"Created RESEARCH node: {research_node_id}")

            except Exception as e:
                logger.warning(f"Failed to create RESEARCH node: {e}")

        if rerun_logger:
            rerun_logger.log_thought(
                "research_similarity",
                f"Similarity search complete: {len(projects)} projects, "
                f"{len(modules)} modules, {len(implementations)} implementations"
            )

        return {
            "projects": projects,
            "modules": modules,
            "implementations": implementations,
            "suggestions": suggestions,
            "research_node_id": research_node_id,
        }

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return {
            "projects": [],
            "modules": [],
            "implementations": [],
            "suggestions": f"Similarity search failed: {str(e)}",
            "research_node_id": None,
        }


def search_similar_code_sync(
    spec: str,
    session_id: str,
    req_node_id: Optional[str] = None,
    rerun_logger: Optional[Any] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for search_similar_code.

    Use this when calling from non-async context (e.g., orchestrator nodes).

    Args:
        spec: User's specification
        session_id: Session ID
        req_node_id: Optional REQ node to link to
        rerun_logger: Optional RerunLogger
        **kwargs: Additional arguments passed to search_similar_code

    Returns:
        Dictionary with similarity findings
    """
    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        # No event loop running, create one
        return asyncio.run(
            search_similar_code(
                spec=spec,
                session_id=session_id,
                req_node_id=req_node_id,
                rerun_logger=rerun_logger,
                **kwargs,
            )
        )
    else:
        # Event loop already running, use run_until_complete on a new loop
        # This is a workaround for nested event loops
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run,
                search_similar_code(
                    spec=spec,
                    session_id=session_id,
                    req_node_id=req_node_id,
                    rerun_logger=rerun_logger,
                    **kwargs,
                )
            )
            return future.result()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def should_run_similarity_search(spec: str, task_category: Optional[str] = None) -> bool:
    """
    Determine if similarity search is worthwhile for this spec.

    Args:
        spec: User's specification
        task_category: Optional task category from ResearchArtifact

    Returns:
        True if similarity search should be run
    """
    # Skip for very short specs (likely too vague)
    if len(spec.split()) < 5:
        return False

    # Skip for specific categories where similarity is less useful
    if task_category in ["debug", "refactor"]:
        return False

    # Run for greenfield, algorithmic, and systems tasks
    return True


def format_similarity_summary_for_spec(
    projects: List,
    modules: List,
    implementations: List,
) -> str:
    """
    Create a brief summary to append to the spec.

    Args:
        projects: List of similar projects
        modules: List of similar modules
        implementations: List of similar implementations

    Returns:
        Markdown-formatted summary
    """
    if not (projects or modules or implementations):
        return ""

    lines = [
        "",
        "# Similar Code Found",
        "",
    ]

    if projects:
        lines.append(f"**Similar Projects:** {len(projects)} found")
        for project in projects[:2]:  # Top 2
            lines.append(f"- {project.name} ({project.similarity_score:.0%}): {project.url}")
        lines.append("")

    if modules:
        lines.append(f"**Similar Modules:** {len(modules)} found")
        for module in modules[:2]:  # Top 2
            lines.append(f"- {module.name} ({module.similarity_score:.0%}): {module.source_url}")
        lines.append("")

    if implementations:
        lines.append(f"**Similar Implementations:** {len(implementations)} found")
        for impl in implementations[:2]:  # Top 2
            lines.append(f"- {impl.function_name} ({impl.similarity_score:.0%}): {impl.source_url}")
        lines.append("")

    lines.append("_Review these before implementing from scratch!_")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_similarity_config() -> Dict[str, Any]:
    """
    Get similarity search configuration.

    Returns:
        Dictionary with configuration settings
    """
    # Check environment variables for configuration
    import os

    return {
        "enabled": os.getenv("PARAGON_SIMILARITY_SEARCH", "true").lower() == "true",
        "project_search": os.getenv("PARAGON_SIMILARITY_PROJECTS", "true").lower() == "true",
        "module_search": os.getenv("PARAGON_SIMILARITY_MODULES", "true").lower() == "true",
        "implementation_search": os.getenv("PARAGON_SIMILARITY_IMPLS", "false").lower() == "true",
        "available": SIMILARITY_AVAILABLE,
    }
