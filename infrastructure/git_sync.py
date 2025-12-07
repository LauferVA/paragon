"""
PARAGON GIT SYNC - The Commit Choreographer

Auto-triggers Git commits on graph transaction boundaries with semantic messages
derived from the Teleology Chain (CODE -> SPEC -> REQ).

Architecture:
- Commit on Transaction Boundaries (Node + Edge creation), NOT on every token
- Semantic Messages: Traverse teleology chain to generate: "feat: [REQ-123] Implement hash function"
- Agent Attribution: Tag commits with agent_id that authored the change
- Hook Pattern: Integrates via callback/hook from add_node_safe

Design Principles:
1. NO hardcoding - all config from paragon.toml
2. msgspec.Struct for all data classes (NO Pydantic)
3. Git operations are ATOMIC - either full transaction commits or nothing
4. Semantic commit messages follow Conventional Commits spec

Usage:
    git_sync = GitSync()
    git_sync.on_transaction_complete(
        nodes_created=["node_123"],
        edges_created=[("node_123", "spec_456", "IMPLEMENTS")],
        agent_id="builder-gpt4-v1"
    )
"""
import msgspec
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import warnings

from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType


# =============================================================================
# CONFIGURATION
# =============================================================================

class GitSyncConfig(msgspec.Struct, kw_only=True, frozen=True):
    """Configuration for Git sync operations."""
    enabled: bool = True                     # Enable/disable git sync
    repo_path: str = "."                     # Path to git repository
    auto_commit: bool = True                 # Auto-commit on transaction boundaries
    auto_push: bool = False                  # Auto-push after commit (dangerous!)
    commit_prefix: str = ""                  # Optional prefix for all commits
    author_name: str = "Paragon"             # Git author name
    author_email: str = "paragon@localhost"  # Git author email


def load_git_config_from_graph(db) -> Optional[GitSyncConfig]:
    """
    Load git sync configuration from graph (graph-native approach).

    Wave 6 Refactor: Config lives in graph, not files.

    Args:
        db: ParagonDB instance with CONFIG nodes

    Returns:
        GitSyncConfig or None if not available
    """
    try:
        from infrastructure.config_graph import get_config
        git_cfg = get_config(db, "git")
        if git_cfg:
            return GitSyncConfig(
                enabled=git_cfg.get("enabled", True),
                repo_path=git_cfg.get("repo_path", "."),
                auto_commit=git_cfg.get("auto_commit", True),
                auto_push=git_cfg.get("auto_push", False),
                commit_prefix=git_cfg.get("commit_prefix", ""),
                author_name=git_cfg.get("author_name", "Paragon"),
                author_email=git_cfg.get("author_email", "paragon@localhost"),
            )
    except Exception:
        pass
    return None


def load_git_config_from_toml() -> GitSyncConfig:
    """
    Load git sync configuration from paragon.toml (legacy fallback).

    Returns:
        GitSyncConfig with settings
    """
    try:
        import tomllib
        config_path = Path(__file__).parent.parent / "config" / "paragon.toml"

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        git_cfg = config.get("git", {})
        return GitSyncConfig(
            enabled=git_cfg.get("enabled", True),
            repo_path=git_cfg.get("repo_path", "."),
            auto_commit=git_cfg.get("auto_commit", True),
            auto_push=git_cfg.get("auto_push", False),
            commit_prefix=git_cfg.get("commit_prefix", ""),
            author_name=git_cfg.get("author_name", "Paragon"),
            author_email=git_cfg.get("author_email", "paragon@localhost"),
        )
    except Exception as e:
        warnings.warn(f"Failed to load git config from TOML: {e}")
        return GitSyncConfig()


def load_git_config(db=None) -> GitSyncConfig:
    """
    Load git sync configuration with graph-native priority.

    Resolution order: Graph -> TOML -> Defaults

    Args:
        db: Optional ParagonDB instance for graph-native config

    Returns:
        GitSyncConfig with settings
    """
    # Try graph-native config first
    if db is not None:
        graph_config = load_git_config_from_graph(db)
        if graph_config:
            return graph_config

    # Fall back to TOML
    return load_git_config_from_toml()


# =============================================================================
# COMMIT MESSAGE GENERATOR
# =============================================================================

class CommitType(msgspec.Struct, frozen=True):
    """Conventional Commit type prefixes."""
    type: str
    description: str


# Conventional Commits types
COMMIT_TYPES = {
    "feat": CommitType(type="feat", description="A new feature"),
    "fix": CommitType(type="fix", description="A bug fix"),
    "docs": CommitType(type="docs", description="Documentation changes"),
    "refactor": CommitType(type="refactor", description="Code refactoring"),
    "test": CommitType(type="test", description="Test additions/changes"),
    "chore": CommitType(type="chore", description="Maintenance tasks"),
}


class TeleologyChain(msgspec.Struct, kw_only=True, frozen=False):
    """Represents a chain from CODE -> SPEC -> REQ."""
    code_node: Optional[NodeData] = None
    spec_node: Optional[NodeData] = None
    req_node: Optional[NodeData] = None

    def to_commit_message(self) -> str:
        """
        Generate semantic commit message from teleology chain.

        Format: <type>: [REQ-ID] <description>
        Example: feat: [REQ-abc123] Implement hash function
        """
        # Determine commit type based on node types in chain
        if self.code_node:
            commit_type = "feat"
        elif self.spec_node:
            commit_type = "docs"
        elif self.req_node:
            commit_type = "chore"
        else:
            commit_type = "chore"

        # Extract REQ ID if available
        req_id = None
        if self.req_node:
            req_id = self.req_node.id[:8]  # Short hash

        # Generate description
        description = "Update graph"
        if self.code_node:
            # Use first line of code content as description
            content_lines = self.code_node.content.strip().split('\n')
            if content_lines:
                description = content_lines[0][:60]  # Limit length
        elif self.spec_node:
            content_lines = self.spec_node.content.strip().split('\n')
            if content_lines:
                description = content_lines[0][:60]
        elif self.req_node:
            content_lines = self.req_node.content.strip().split('\n')
            if content_lines:
                description = content_lines[0][:60]

        # Construct message
        if req_id:
            return f"{commit_type}: [REQ-{req_id}] {description}"
        else:
            return f"{commit_type}: {description}"


# =============================================================================
# GIT OPERATIONS
# =============================================================================

class GitSync:
    """
    Git synchronization manager.

    Handles automatic Git commits on transaction boundaries with semantic
    messages derived from the teleology chain.

    Thread-safety: NOT thread-safe. External locking required for concurrent use.
    """

    def __init__(self, config: Optional[GitSyncConfig] = None, db=None):
        """
        Initialize Git sync manager.

        Args:
            config: Optional GitSyncConfig. If None, loads from paragon.toml
            db: Optional ParagonDB instance for teleology traversal
        """
        self.config = config or load_git_config()
        self.db = db
        self.repo_path = Path(self.config.repo_path)

        # Validate git repository
        if self.config.enabled:
            self._ensure_git_repo()

    def _ensure_git_repo(self) -> None:
        """Ensure we're in a git repository."""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            warnings.warn(
                f"Not a git repository: {self.repo_path}. "
                "Git sync will be disabled."
            )
            # Modify config to disable
            object.__setattr__(self.config, 'enabled', False)

    def _run_git_command(
        self,
        args: List[str],
        capture_output: bool = True,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Run a git command.

        Args:
            args: Command arguments (e.g., ["add", "."])
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise on non-zero exit

        Returns:
            CompletedProcess result

        Raises:
            subprocess.CalledProcessError: If command fails and check=True
        """
        cmd = ["git"] + args
        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            capture_output=capture_output,
            text=True,
            check=check,
            env={
                **subprocess.os.environ,
                "GIT_AUTHOR_NAME": self.config.author_name,
                "GIT_AUTHOR_EMAIL": self.config.author_email,
                "GIT_COMMITTER_NAME": self.config.author_name,
                "GIT_COMMITTER_EMAIL": self.config.author_email,
            }
        )
        return result

    def _get_teleology_chain(self, node_id: str) -> TeleologyChain:
        """
        Traverse teleology chain from a node to find CODE -> SPEC -> REQ.

        Args:
            node_id: Starting node ID

        Returns:
            TeleologyChain with discovered nodes
        """
        chain = TeleologyChain()

        if not self.db:
            return chain

        try:
            # Get the starting node
            node = self.db.get_node(node_id)

            # Categorize the starting node
            if node.type == NodeType.CODE.value:
                chain.code_node = node
            elif node.type == NodeType.SPEC.value:
                chain.spec_node = node
            elif node.type == NodeType.REQ.value:
                chain.req_node = node

            # Traverse to find missing pieces
            visited = {node_id}
            queue = [node_id]

            while queue and (not all([chain.code_node, chain.spec_node, chain.req_node])):
                current_id = queue.pop(0)

                # Get outgoing edges (what this node points to)
                outgoing = self.db.get_outgoing_edges(current_id)
                for edge_dict in outgoing:
                    target_id = edge_dict["target"]
                    if target_id in visited:
                        continue

                    visited.add(target_id)
                    target_node = self.db.get_node(target_id)

                    # Categorize target
                    if target_node.type == NodeType.SPEC.value and not chain.spec_node:
                        chain.spec_node = target_node
                        queue.append(target_id)
                    elif target_node.type == NodeType.REQ.value and not chain.req_node:
                        chain.req_node = target_node

        except Exception as e:
            warnings.warn(f"Failed to traverse teleology chain: {e}")

        return chain

    def on_transaction_complete(
        self,
        nodes_created: List[str],
        edges_created: List[Tuple[str, str, str]],
        agent_id: Optional[str] = None,
        agent_role: Optional[str] = None,
    ) -> bool:
        """
        Called when a graph transaction completes.

        This is the main hook called by ParagonDB or orchestrator after
        a logical transaction (e.g., creating a CODE node + IMPLEMENTS edge).

        Args:
            nodes_created: List of node IDs created in this transaction
            edges_created: List of (source_id, target_id, edge_type) tuples
            agent_id: Optional agent ID that performed the transaction
            agent_role: Optional agent role (e.g., "BUILDER")

        Returns:
            True if commit succeeded, False otherwise
        """
        if not self.config.enabled or not self.config.auto_commit:
            return False

        if not nodes_created and not edges_created:
            # Empty transaction, nothing to commit
            return False

        try:
            # Generate commit message based on teleology
            if nodes_created and self.db:
                # Use first created node to derive message
                primary_node_id = nodes_created[0]
                chain = self._get_teleology_chain(primary_node_id)
                message = chain.to_commit_message()
            else:
                # Fallback message
                message = f"chore: Update graph ({len(nodes_created)} nodes, {len(edges_created)} edges)"

            # Add prefix if configured
            if self.config.commit_prefix:
                message = f"{self.config.commit_prefix}{message}"

            # Add agent attribution
            footer = "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n"
            footer += "Co-Authored-By: Claude <noreply@anthropic.com>\n"
            if agent_id:
                footer += f"Agent-ID: {agent_id}\n"
            if agent_role:
                footer += f"Agent-Role: {agent_role}\n"

            full_message = message + footer

            # Stage all changes
            self._run_git_command(["add", "."])

            # Check if there are changes to commit
            status_result = self._run_git_command(["status", "--porcelain"])
            if not status_result.stdout.strip():
                # No changes to commit
                return False

            # Commit
            self._run_git_command(["commit", "-m", full_message])

            # Optionally push (dangerous in production!)
            if self.config.auto_push:
                warnings.warn("Auto-push is enabled. This may overwrite remote changes!")
                try:
                    self._run_git_command(["push"], check=False)
                except Exception as e:
                    warnings.warn(f"Failed to push: {e}")

            return True

        except subprocess.CalledProcessError as e:
            warnings.warn(f"Git commit failed: {e.stderr if e.stderr else str(e)}")
            return False
        except Exception as e:
            warnings.warn(f"Git sync error: {e}")
            return False

    def create_tag(self, tag_name: str, message: Optional[str] = None) -> bool:
        """
        Create a git tag at the current commit.

        Args:
            tag_name: Name of the tag (e.g., "v1.0.0")
            message: Optional tag message

        Returns:
            True if tag created successfully
        """
        if not self.config.enabled:
            return False

        try:
            if message:
                self._run_git_command(["tag", "-a", tag_name, "-m", message])
            else:
                self._run_git_command(["tag", tag_name])
            return True
        except Exception as e:
            warnings.warn(f"Failed to create tag {tag_name}: {e}")
            return False

    def get_current_commit(self) -> Optional[str]:
        """
        Get the current commit hash.

        Returns:
            Commit hash (short) or None if not in a repo
        """
        if not self.config.enabled:
            return None

        try:
            result = self._run_git_command(["rev-parse", "--short", "HEAD"])
            return result.stdout.strip()
        except Exception:
            return None

    def get_commit_count(self) -> int:
        """
        Get total number of commits in the repository.

        Returns:
            Number of commits, or 0 if error
        """
        if not self.config.enabled:
            return 0

        try:
            result = self._run_git_command(["rev-list", "--count", "HEAD"])
            return int(result.stdout.strip())
        except Exception:
            return 0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_git_sync: Optional[GitSync] = None


def get_git_sync(db=None) -> GitSync:
    """Get or create the global GitSync instance."""
    global _global_git_sync
    if _global_git_sync is None:
        _global_git_sync = GitSync(db=db)
    elif db and not _global_git_sync.db:
        # Update db reference if provided
        _global_git_sync.db = db
    return _global_git_sync


def on_transaction_complete(
    nodes_created: List[str],
    edges_created: List[Tuple[str, str, str]],
    agent_id: Optional[str] = None,
    agent_role: Optional[str] = None,
    db=None
) -> bool:
    """Convenience function to trigger git sync on transaction complete."""
    sync = get_git_sync(db=db)
    return sync.on_transaction_complete(
        nodes_created=nodes_created,
        edges_created=edges_created,
        agent_id=agent_id,
        agent_role=agent_role
    )
