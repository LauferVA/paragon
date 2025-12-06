"""
PARAGON DOCUMENTER - The Scribe Agent

Background agent that generates documentation from the graph state,
not from prompts. It reflects the GRAPH, not the INTENT.

Responsibilities:
1. Auto-README: Complete rewrite of README.md on every major cycle
2. The Wiki: Generate Markdown files in docs/wiki/ for high-level SPEC nodes
3. Changelog: Append to CHANGELOG.md by diffing Merkle Root between commits

Design Principles:
1. NO prompts - query the graph only
2. Complete overwrite, no merge (README)
3. Append-only for changelog
4. msgspec.Struct for all data classes (NO Pydantic)
5. Background agent - runs async, doesn't block main orchestration

Architecture:
- Queries ParagonDB for REQ, SPEC, CODE nodes
- Uses teleology chain to understand structure
- Generates markdown from node content directly
- Integrates with GitSync for Merkle diff detection

Usage:
    documenter = Documenter(db=paragon_db)
    documenter.generate_readme()
    documenter.generate_wiki()
    documenter.append_changelog(old_merkle="abc123", new_merkle="def456")
"""
import msgspec
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
from datetime import datetime, timezone
import warnings

from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# CONFIGURATION
# =============================================================================

class DocumenterConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration for Documenter agent."""
    readme_path: str = "README.md"
    changelog_path: str = "CHANGELOG.md"
    wiki_path: str = "docs/wiki"
    auto_generate: bool = True
    include_pending_nodes: bool = False  # Whether to document pending nodes


def load_documenter_config() -> DocumenterConfig:
    """
    Load documenter configuration from paragon.toml.

    Returns:
        DocumenterConfig with settings
    """
    try:
        import tomllib
        config_path = Path(__file__).parent.parent / "config" / "paragon.toml"

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        doc_cfg = config.get("documenter", {})
        return DocumenterConfig(
            readme_path=doc_cfg.get("readme_path", "README.md"),
            changelog_path=doc_cfg.get("changelog_path", "CHANGELOG.md"),
            wiki_path=doc_cfg.get("wiki_path", "docs/wiki"),
            auto_generate=doc_cfg.get("auto_generate", True),
            include_pending_nodes=doc_cfg.get("include_pending_nodes", False),
        )
    except Exception as e:
        warnings.warn(f"Failed to load documenter config, using defaults: {e}")
        return DocumenterConfig()


# =============================================================================
# MARKDOWN GENERATORS
# =============================================================================

class MarkdownBuilder:
    """Helper class for building markdown documents."""

    def __init__(self):
        self.lines: List[str] = []

    def h1(self, text: str) -> "MarkdownBuilder":
        """Add H1 heading."""
        self.lines.append(f"# {text}\n")
        return self

    def h2(self, text: str) -> "MarkdownBuilder":
        """Add H2 heading."""
        self.lines.append(f"## {text}\n")
        return self

    def h3(self, text: str) -> "MarkdownBuilder":
        """Add H3 heading."""
        self.lines.append(f"### {text}\n")
        return self

    def h4(self, text: str) -> "MarkdownBuilder":
        """Add H4 heading."""
        self.lines.append(f"#### {text}\n")
        return self

    def p(self, text: str) -> "MarkdownBuilder":
        """Add paragraph."""
        self.lines.append(f"{text}\n")
        return self

    def code_block(self, code: str, language: str = "") -> "MarkdownBuilder":
        """Add code block."""
        self.lines.append(f"```{language}\n{code}\n```\n")
        return self

    def ul(self, items: List[str]) -> "MarkdownBuilder":
        """Add unordered list."""
        for item in items:
            self.lines.append(f"- {item}\n")
        self.lines.append("\n")
        return self

    def ol(self, items: List[str]) -> "MarkdownBuilder":
        """Add ordered list."""
        for i, item in enumerate(items, 1):
            self.lines.append(f"{i}. {item}\n")
        self.lines.append("\n")
        return self

    def table(self, headers: List[str], rows: List[List[str]]) -> "MarkdownBuilder":
        """Add markdown table."""
        # Header row
        self.lines.append("| " + " | ".join(headers) + " |\n")
        # Separator row
        self.lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
        # Data rows
        for row in rows:
            self.lines.append("| " + " | ".join(row) + " |\n")
        self.lines.append("\n")
        return self

    def hr(self) -> "MarkdownBuilder":
        """Add horizontal rule."""
        self.lines.append("---\n\n")
        return self

    def newline(self) -> "MarkdownBuilder":
        """Add blank line."""
        self.lines.append("\n")
        return self

    def to_string(self) -> str:
        """Convert to markdown string."""
        return "".join(self.lines)


# =============================================================================
# DOCUMENTER AGENT
# =============================================================================

class Documenter:
    """
    Documentation generator agent.

    Generates documentation from graph state, not from prompts.
    Reflects what IS, not what SHOULD BE.
    """

    def __init__(self, db=None, config: Optional[DocumenterConfig] = None):
        """
        Initialize the Documenter agent.

        Args:
            db: ParagonDB instance
            config: Optional DocumenterConfig
        """
        self.db = db
        self.config = config or load_documenter_config()

    def generate_readme(self) -> bool:
        """
        Generate README.md from current graph state.

        Completely overwrites existing README.md with content derived
        from REQ and SPEC nodes.

        Returns:
            True if successful, False otherwise
        """
        if not self.db:
            warnings.warn("Cannot generate README: no database provided")
            return False

        try:
            md = MarkdownBuilder()

            # Header
            md.h1("Project Paragon")
            md.p("*Graph-native AI software platform - Auto-generated from graph state*")
            md.newline()

            # Get all REQ nodes
            req_nodes = self.db.get_nodes_by_type(NodeType.REQ.value)

            if not req_nodes:
                md.h2("Status")
                md.p("No requirements defined yet.")
            else:
                # Overview section
                md.h2("Overview")
                md.p(f"This project contains {len(req_nodes)} requirement(s):")
                md.newline()

                # List requirements
                for req in req_nodes:
                    status_emoji = self._get_status_emoji(req.status)
                    preview = self._get_content_preview(req.content, 100)
                    md.p(f"{status_emoji} **REQ-{req.id[:8]}**: {preview}")

                md.newline()

                # Specifications section
                md.h2("Specifications")
                spec_nodes = self.db.get_nodes_by_type(NodeType.SPEC.value)

                if spec_nodes:
                    # Filter to verified/processing specs
                    active_specs = [
                        s for s in spec_nodes
                        if s.status in [NodeStatus.VERIFIED.value, NodeStatus.PROCESSING.value]
                        or self.config.include_pending_nodes
                    ]

                    if active_specs:
                        rows = []
                        for spec in active_specs:
                            status = self._get_status_emoji(spec.status)
                            preview = self._get_content_preview(spec.content, 60)
                            # Find parent REQ
                            parent_req = self._find_parent_req(spec.id)
                            req_ref = f"REQ-{parent_req[:8]}" if parent_req else "N/A"
                            rows.append([f"SPEC-{spec.id[:8]}", status, preview, req_ref])

                        md.table(
                            headers=["ID", "Status", "Description", "Requirement"],
                            rows=rows
                        )
                    else:
                        md.p("No active specifications.")
                else:
                    md.p("No specifications defined.")

                md.newline()

                # Implementation section
                md.h2("Implementation")
                code_nodes = self.db.get_nodes_by_type(NodeType.CODE.value)

                if code_nodes:
                    active_code = [
                        c for c in code_nodes
                        if c.status in [NodeStatus.VERIFIED.value, NodeStatus.TESTED.value]
                        or self.config.include_pending_nodes
                    ]

                    if active_code:
                        md.p(f"Implemented modules: {len(active_code)}")
                        md.newline()

                        # Group by file path if available
                        file_groups: Dict[str, List[NodeData]] = {}
                        for code in active_code:
                            file_path = code.data.get("file_path", "unknown")
                            if file_path not in file_groups:
                                file_groups[file_path] = []
                            file_groups[file_path].append(code)

                        for file_path, codes in sorted(file_groups.items()):
                            md.h3(f"`{file_path}`")
                            for code in codes:
                                status = self._get_status_emoji(code.status)
                                preview = self._get_content_preview(code.content, 80)
                                md.p(f"{status} {preview}")
                            md.newline()
                    else:
                        md.p("No verified implementations yet.")
                else:
                    md.p("No code implemented.")

            # Footer
            md.hr()
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            md.p(f"*Auto-generated by Paragon Documenter on {timestamp}*")
            md.p(f"*Graph contains {self.db.node_count} nodes and {self.db.edge_count} edges*")

            # Write to file
            readme_path = Path(self.config.readme_path)
            readme_path.write_text(md.to_string(), encoding="utf-8")

            return True

        except Exception as e:
            warnings.warn(f"Failed to generate README: {e}")
            return False

    def generate_wiki(self) -> bool:
        """
        Generate wiki pages for high-level SPEC nodes.

        Creates individual markdown files in docs/wiki/ for each
        verified SPEC node.

        Returns:
            True if successful, False otherwise
        """
        if not self.db:
            warnings.warn("Cannot generate wiki: no database provided")
            return False

        try:
            wiki_path = Path(self.config.wiki_path)
            wiki_path.mkdir(parents=True, exist_ok=True)

            # Get verified SPEC nodes
            spec_nodes = self.db.get_nodes_by_type(NodeType.SPEC.value)
            verified_specs = [
                s for s in spec_nodes
                if s.status == NodeStatus.VERIFIED.value
            ]

            if not verified_specs:
                warnings.warn("No verified specs to document")
                return False

            # Generate a page for each spec
            for spec in verified_specs:
                md = MarkdownBuilder()

                # Title
                spec_title = self._get_content_preview(spec.content, 60)
                md.h1(f"SPEC-{spec.id[:8]}: {spec_title}")
                md.newline()

                # Metadata
                md.h2("Metadata")
                md.ul([
                    f"**Status**: {spec.status}",
                    f"**Created**: {spec.created_at}",
                    f"**Created By**: {spec.created_by}",
                    f"**Version**: {spec.version}",
                ])

                # Content
                md.h2("Specification")
                md.p(spec.content)
                md.newline()

                # Find parent REQ
                parent_req_id = self._find_parent_req(spec.id)
                if parent_req_id:
                    parent_req = self.db.get_node(parent_req_id)
                    md.h2("Requirement")
                    md.p(f"**REQ-{parent_req.id[:8]}**")
                    md.p(parent_req.content)
                    md.newline()

                # Find implementing CODE
                implementing_code = self._find_implementing_code(spec.id)
                if implementing_code:
                    md.h2("Implementation")
                    for code in implementing_code:
                        md.h3(f"CODE-{code.id[:8]}")
                        file_path = code.data.get("file_path", "N/A")
                        md.p(f"**File**: `{file_path}`")
                        md.code_block(code.content, language="python")

                # Write file
                filename = f"spec_{spec.id[:8]}.md"
                filepath = wiki_path / filename
                filepath.write_text(md.to_string(), encoding="utf-8")

            return True

        except Exception as e:
            warnings.warn(f"Failed to generate wiki: {e}")
            return False

    def append_changelog(
        self,
        old_merkle: Optional[str],
        new_merkle: str,
        description: Optional[str] = None
    ) -> bool:
        """
        Append to CHANGELOG.md by diffing Merkle root between commits.

        Args:
            old_merkle: Previous Merkle root hash (None if first entry)
            new_merkle: Current Merkle root hash
            description: Optional description of changes

        Returns:
            True if successful, False otherwise
        """
        try:
            changelog_path = Path(self.config.changelog_path)

            # Ensure file exists
            if not changelog_path.exists():
                # Create initial changelog
                initial_content = "# Changelog\n\n"
                initial_content += "All notable changes to this project will be documented in this file.\n\n"
                initial_content += "The format is based on graph Merkle root changes.\n\n"
                changelog_path.write_text(initial_content, encoding="utf-8")

            # Read existing content
            existing_content = changelog_path.read_text(encoding="utf-8")

            # Build new entry
            md = MarkdownBuilder()
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            md.h2(f"[{new_merkle[:8]}] - {timestamp}")

            if description:
                md.p(description)

            if old_merkle:
                md.p(f"**Merkle Diff**: `{old_merkle[:8]}` → `{new_merkle[:8]}`")
            else:
                md.p(f"**Merkle Root**: `{new_merkle[:8]}` (initial)")

            # Add graph stats if db available
            if self.db:
                md.p(f"**Graph State**: {self.db.node_count} nodes, {self.db.edge_count} edges")

            md.newline()

            # Append to changelog (prepend after header)
            lines = existing_content.split('\n')
            header_end = 0
            for i, line in enumerate(lines):
                if line.startswith('##'):
                    header_end = i
                    break

            if header_end == 0:
                # No existing entries, append to end
                new_content = existing_content + "\n" + md.to_string()
            else:
                # Insert before first entry
                new_content = '\n'.join(lines[:header_end]) + "\n\n" + md.to_string() + '\n'.join(lines[header_end:])

            changelog_path.write_text(new_content, encoding="utf-8")
            return True

        except Exception as e:
            warnings.warn(f"Failed to append changelog: {e}")
            return False

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_status_emoji(self, status: str) -> str:
        """Get emoji representation of node status."""
        status_map = {
            NodeStatus.VERIFIED.value: "✅",
            NodeStatus.TESTED.value: "🧪",
            NodeStatus.PROCESSING.value: "⏳",
            NodeStatus.PENDING.value: "⏸️",
            NodeStatus.FAILED.value: "❌",
            NodeStatus.BLOCKED.value: "🚫",
        }
        return status_map.get(status, "❓")

    def _get_content_preview(self, content: str, max_length: int = 100) -> str:
        """Get preview of content (first line, truncated)."""
        if not content:
            return "(empty)"

        # Get first non-empty line
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return "(empty)"

        first_line = lines[0]
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."
        return first_line

    def _find_parent_req(self, node_id: str) -> Optional[str]:
        """
        Find parent REQ for a node by traversing teleology chain.

        Args:
            node_id: Node to find parent for

        Returns:
            REQ node ID or None
        """
        if not self.db:
            return None

        try:
            # Traverse outgoing edges looking for REQ
            visited = {node_id}
            queue = [node_id]

            while queue:
                current_id = queue.pop(0)
                current_node = self.db.get_node(current_id)

                # Check if this is a REQ
                if current_node.type == NodeType.REQ.value:
                    return current_id

                # Traverse outgoing edges (TRACES_TO, IMPLEMENTS, etc.)
                outgoing = self.db.get_outgoing_edges(current_id)
                for edge_dict in outgoing:
                    target_id = edge_dict["target"]
                    if target_id not in visited:
                        visited.add(target_id)
                        queue.append(target_id)

            return None

        except Exception:
            return None

    def _find_implementing_code(self, spec_id: str) -> List[NodeData]:
        """
        Find CODE nodes that implement a SPEC.

        Args:
            spec_id: SPEC node ID

        Returns:
            List of CODE nodes implementing this spec
        """
        if not self.db:
            return []

        try:
            # Find incoming IMPLEMENTS edges
            implementing = []
            incoming = self.db.get_incoming_edges(spec_id)

            for edge_dict in incoming:
                if edge_dict["type"] == EdgeType.IMPLEMENTS.value:
                    source_id = edge_dict["source"]
                    code_node = self.db.get_node(source_id)
                    if code_node.type == NodeType.CODE.value:
                        implementing.append(code_node)

            return implementing

        except Exception:
            return []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_documenter: Optional[Documenter] = None


def get_documenter(db=None) -> Documenter:
    """Get or create the global Documenter instance."""
    global _global_documenter
    if _global_documenter is None:
        _global_documenter = Documenter(db=db)
    elif db and not _global_documenter.db:
        _global_documenter.db = db
    return _global_documenter


def generate_all_docs(db=None) -> Dict[str, bool]:
    """
    Generate all documentation.

    Returns:
        Dict with success status for each doc type
    """
    documenter = get_documenter(db=db)
    return {
        "readme": documenter.generate_readme(),
        "wiki": documenter.generate_wiki(),
    }
