"""
PARAGON SIMILARITY ADVISOR - User-Facing Suggestions

Formats similarity search results into actionable suggestions for the user.
Helps determine when similarity is high enough to suggest alternatives.

Design Principles:
1. CLEAR COMMUNICATION: Format suggestions in user-friendly language
2. CONTEXTUAL THRESHOLDS: Different similarity thresholds for different levels
3. ACTIONABLE: Always provide clear next steps
4. TRANSPARENT: Explain why we're suggesting something
"""

import msgspec
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Import similarity types
try:
    from infrastructure.repo_scanner import SimilarProject, SimilarModule, SimilarImplementation
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False


# =============================================================================
# SIMILARITY ADVISOR
# =============================================================================

class SimilarityAdvisor:
    """
    Advises user about similar existing code.

    Provides formatted suggestions with context-appropriate thresholds:
    - Projects: 0.7+ = strong match, 0.5-0.7 = possible alternative
    - Modules: 0.7+ = drop-in candidate, 0.6-0.7 = adapter needed
    - Implementations: 0.8+ = exact match, 0.7-0.8 = similar approach
    """

    def __init__(self):
        """Initialize the similarity advisor."""
        # Thresholds for different levels
        self.project_threshold_strong = 0.7
        self.project_threshold_weak = 0.5

        self.module_threshold_dropin = 0.7
        self.module_threshold_adapter = 0.6

        self.implementation_threshold_exact = 0.8
        self.implementation_threshold_similar = 0.7

    def format_project_suggestion(
        self,
        project,  # SimilarProject
        user_spec: str,
    ) -> str:
        """
        Format a suggestion message for whole-project match.

        Args:
            project: SimilarProject object
            user_spec: User's project specification

        Returns:
            Formatted suggestion message
        """
        score = project.similarity_score

        # Determine suggestion strength
        if score >= self.project_threshold_strong:
            strength = "STRONG MATCH"
            action = "You might want to use this instead of building from scratch"
        elif score >= self.project_threshold_weak:
            strength = "Possible Alternative"
            action = "Consider reviewing this for ideas or components"
        else:
            strength = "Weak Match"
            action = "Listed for reference, but probably not a direct alternative"

        # Build message
        lines = [
            f"📦 {strength}: {project.name}",
            f"   Similarity: {score:.0%}",
            f"   URL: {project.url}",
        ]

        if project.description:
            lines.append(f"   Description: {project.description[:150]}")

        if project.stars:
            lines.append(f"   Stars: {project.stars:,}")

        if project.language:
            lines.append(f"   Language: {project.language}")

        if project.tech_stack:
            lines.append(f"   Tech Stack: {', '.join(project.tech_stack[:3])}")

        lines.append(f"   → {action}")
        lines.append("")

        return "\n".join(lines)

    def format_module_suggestion(
        self,
        module,  # SimilarModule
        user_module_spec: str,
    ) -> str:
        """
        Format a suggestion for reusable module.

        Args:
            module: SimilarModule object
            user_module_spec: User's module specification

        Returns:
            Formatted suggestion message
        """
        score = module.similarity_score

        # Determine integration complexity
        if score >= self.module_threshold_dropin:
            complexity = "Drop-in Candidate"
            action = "This module might work with minimal changes"
            icon = "✅"
        elif score >= self.module_threshold_adapter:
            complexity = "Adapter Needed"
            action = "You could wrap this module with an adapter layer"
            icon = "🔧"
        else:
            complexity = "Inspiration Only"
            action = "Review the approach but expect to rewrite"
            icon = "💡"

        lines = [
            f"{icon} {complexity}: {module.name}",
            f"   Similarity: {score:.0%}",
            f"   Source: {module.source_url}",
        ]

        if module.description:
            lines.append(f"   Description: {module.description[:150]}")

        if module.file_path:
            lines.append(f"   Path: {module.file_path}")

        if module.dependencies:
            lines.append(f"   Dependencies: {', '.join(module.dependencies[:3])}")

        lines.append(f"   → {action}")
        lines.append("")

        return "\n".join(lines)

    def format_implementation_suggestion(
        self,
        implementation,  # SimilarImplementation
        user_task: str,
    ) -> str:
        """
        Format a suggestion for specific implementation.

        Args:
            implementation: SimilarImplementation object
            user_task: User's task description

        Returns:
            Formatted suggestion message
        """
        score = implementation.similarity_score

        # Determine match quality
        if score >= self.implementation_threshold_exact:
            quality = "Exact Match"
            action = "This implementation solves your exact problem"
            icon = "🎯"
        elif score >= self.implementation_threshold_similar:
            quality = "Similar Approach"
            action = "Review this implementation for the pattern/algorithm"
            icon = "🔍"
        else:
            quality = "Related"
            action = "Might provide useful context"
            icon = "📚"

        lines = [
            f"{icon} {quality}: {implementation.function_name}",
            f"   Similarity: {score:.0%}",
            f"   Source: {implementation.source_url}",
        ]

        if implementation.description:
            lines.append(f"   Description: {implementation.description[:150]}")

        if implementation.license:
            lines.append(f"   License: {implementation.license}")

        if implementation.tags:
            lines.append(f"   Tags: {', '.join(implementation.tags[:3])}")

        # Show code snippet if available
        if implementation.code:
            lines.append("\n   Code Preview:")
            code_lines = implementation.code.split("\n")[:10]  # First 10 lines
            for line in code_lines:
                lines.append(f"   │ {line}")
            if len(implementation.code.split("\n")) > 10:
                lines.append("   │ ...")

        lines.append(f"\n   → {action}")
        lines.append("")

        return "\n".join(lines)

    def should_suggest_project(self, similarity_score: float) -> bool:
        """
        Determine if project similarity is high enough to suggest.

        Args:
            similarity_score: Similarity score (0.0-1.0)

        Returns:
            True if should suggest to user
        """
        return similarity_score >= self.project_threshold_weak

    def should_suggest_module(self, similarity_score: float) -> bool:
        """
        Determine if module similarity is high enough to suggest.

        Args:
            similarity_score: Similarity score (0.0-1.0)

        Returns:
            True if should suggest to user
        """
        return similarity_score >= self.module_threshold_adapter

    def should_suggest_implementation(self, similarity_score: float) -> bool:
        """
        Determine if implementation similarity is high enough to suggest.

        Args:
            similarity_score: Similarity score (0.0-1.0)

        Returns:
            True if should suggest to user
        """
        return similarity_score >= self.implementation_threshold_similar

    def format_summary(
        self,
        projects: List = [],  # List[SimilarProject]
        modules: List = [],  # List[SimilarModule]
        implementations: List = [],  # List[SimilarImplementation]
    ) -> str:
        """
        Format a summary of all similarity findings.

        Args:
            projects: List of similar projects
            modules: List of similar modules
            implementations: List of similar implementations

        Returns:
            Formatted summary message
        """
        lines = ["=" * 70]
        lines.append("🔍 SIMILARITY SEARCH RESULTS")
        lines.append("=" * 70)
        lines.append("")

        # Projects section
        if projects:
            lines.append(f"📦 Similar Projects Found: {len(projects)}")
            lines.append("-" * 70)
            for project in projects[:3]:  # Top 3
                lines.append(self.format_project_suggestion(project, ""))
        else:
            lines.append("📦 No similar projects found")
            lines.append("")

        # Modules section
        if modules:
            lines.append(f"🔧 Similar Modules Found: {len(modules)}")
            lines.append("-" * 70)
            for module in modules[:3]:  # Top 3
                lines.append(self.format_module_suggestion(module, ""))
        else:
            lines.append("🔧 No similar modules found")
            lines.append("")

        # Implementations section
        if implementations:
            lines.append(f"🎯 Similar Implementations Found: {len(implementations)}")
            lines.append("-" * 70)
            for impl in implementations[:3]:  # Top 3
                lines.append(self.format_implementation_suggestion(impl, ""))
        else:
            lines.append("🎯 No similar implementations found")
            lines.append("")

        # Summary
        total = len(projects) + len(modules) + len(implementations)
        if total > 0:
            lines.append("=" * 70)
            lines.append(f"💡 Total: {total} similar items found across all levels")
            lines.append("   Review these before building from scratch!")
            lines.append("=" * 70)
        else:
            lines.append("=" * 70)
            lines.append("💡 No similar code found. You're breaking new ground!")
            lines.append("=" * 70)

        return "\n".join(lines)

    def format_research_node_content(
        self,
        query: str,
        projects: List = [],
        modules: List = [],
        implementations: List = [],
    ) -> str:
        """
        Format findings for storage in a RESEARCH node.

        Args:
            query: Original search query
            projects: List of similar projects
            modules: List of similar modules
            implementations: List of similar implementations

        Returns:
            Markdown-formatted content for RESEARCH node
        """
        lines = [
            f"# Similarity Search Results",
            f"",
            f"**Query:** {query}",
            f"**Timestamp:** {self._get_timestamp()}",
            f"",
            f"## Summary",
            f"",
            f"- Projects: {len(projects)}",
            f"- Modules: {len(modules)}",
            f"- Implementations: {len(implementations)}",
            f"",
        ]

        # Projects
        if projects:
            lines.append("## Similar Projects")
            lines.append("")
            for i, project in enumerate(projects, 1):
                lines.append(f"### {i}. {project.name} ({project.similarity_score:.0%})")
                lines.append(f"- **URL:** {project.url}")
                if project.description:
                    lines.append(f"- **Description:** {project.description}")
                if project.stars:
                    lines.append(f"- **Stars:** {project.stars:,}")
                lines.append("")

        # Modules
        if modules:
            lines.append("## Similar Modules")
            lines.append("")
            for i, module in enumerate(modules, 1):
                lines.append(f"### {i}. {module.name} ({module.similarity_score:.0%})")
                lines.append(f"- **Source:** {module.source_url}")
                lines.append(f"- **Integration:** {module.integration_complexity}")
                if module.description:
                    lines.append(f"- **Description:** {module.description}")
                lines.append("")

        # Implementations
        if implementations:
            lines.append("## Similar Implementations")
            lines.append("")
            for i, impl in enumerate(implementations, 1):
                lines.append(f"### {i}. {impl.function_name} ({impl.similarity_score:.0%})")
                lines.append(f"- **Source:** {impl.source_url}")
                if impl.license:
                    lines.append(f"- **License:** {impl.license}")
                if impl.code:
                    lines.append(f"- **Code:**")
                    lines.append(f"```{impl.language}")
                    lines.append(impl.code[:500])  # Limit size
                    lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_advisor() -> SimilarityAdvisor:
    """
    Create a SimilarityAdvisor instance.

    Returns:
        SimilarityAdvisor instance
    """
    return SimilarityAdvisor()
