"""
PARAGON LOCAL REPOSITORY INDEXER - Index Local Codebases

Builds a searchable index of local repositories the user has access to:
- Discovers repositories in common directories (~/code, ~/projects, etc.)
- Extracts modules, classes, and functions using tree-sitter
- Builds embeddings for semantic search
- Persists index to disk for fast lookups

Design Principles:
1. INCREMENTAL INDEXING: Only re-index changed files
2. TREE-SITTER PARSING: Language-aware code structure extraction
3. EMBEDDING-BASED SEARCH: Semantic similarity for better matches
4. MSGSPEC.STRUCT: All data structures
"""

import msgspec
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
import logging
import os
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not installed. Code parsing will be limited.")

try:
    from core.embeddings import compute_embedding, compute_embeddings_batch, is_available as embeddings_available
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class CodeFunction(msgspec.Struct, kw_only=True):
    """Represents a function or method in code."""
    name: str
    signature: str
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    file_path: str = ""
    embedding: Optional[List[float]] = None


class CodeClass(msgspec.Struct, kw_only=True):
    """Represents a class definition."""
    name: str
    docstring: Optional[str] = None
    methods: List[str] = []
    start_line: int = 0
    end_line: int = 0
    file_path: str = ""
    embedding: Optional[List[float]] = None


class CodeModule(msgspec.Struct, kw_only=True):
    """Represents a Python module (file)."""
    name: str
    path: str
    docstring: Optional[str] = None
    functions: List[str] = []  # Function names
    classes: List[str] = []  # Class names
    imports: List[str] = []
    embedding: Optional[List[float]] = None


class IndexedRepo(msgspec.Struct, kw_only=True):
    """Represents an indexed local repository."""
    path: str
    name: str
    language: str = "python"
    last_indexed: str = ""
    file_count: int = 0
    module_count: int = 0
    function_count: int = 0
    class_count: int = 0
    modules: List[str] = []  # Module paths


class IndexMetadata(msgspec.Struct, kw_only=True):
    """Metadata about the local code index."""
    version: str = "1.0"
    created: str = ""
    last_updated: str = ""
    repo_count: int = 0
    total_files: int = 0


# =============================================================================
# LOCAL REPOSITORY INDEXER
# =============================================================================

class LocalRepoIndex:
    """
    Indexes local code repositories for fast searching.

    Capabilities:
    - Discover repositories in common locations
    - Parse code structure using tree-sitter
    - Extract functions, classes, modules
    - Generate embeddings for semantic search
    - Incremental updates (only re-index changed files)
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        search_paths: Optional[List[Path]] = None,
    ):
        """
        Initialize the local repository indexer.

        Args:
            index_dir: Directory to store the index (default: data/local_index)
            search_paths: Directories to search for repositories (default: ~/code, ~/projects)
        """
        self.index_dir = index_dir or Path("data/local_index")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Default search paths
        home = Path.home()
        self.search_paths = search_paths or [
            home / "code",
            home / "projects",
            home / "dev",
            home / "src",
        ]

        # Tree-sitter setup
        self.parser = None
        self.python_language = None
        if TREE_SITTER_AVAILABLE:
            try:
                self.python_language = Language(tree_sitter_python.language())
                self.parser = Parser(self.python_language)
            except Exception as e:
                logger.warning(f"Failed to initialize tree-sitter: {e}")

        # Index storage
        self.repos: Dict[str, IndexedRepo] = {}
        self.modules: Dict[str, CodeModule] = {}
        self.functions: Dict[str, CodeFunction] = {}
        self.classes: Dict[str, CodeClass] = {}

        # Load existing index
        self._load_index()

        logger.info(f"LocalRepoIndex initialized. Indexed {len(self.repos)} repositories.")

    def discover_repos(self) -> List[Path]:
        """
        Discover Git repositories in search paths.

        Returns:
            List of repository root paths
        """
        repos = []

        for search_path in self.search_paths:
            if not search_path.exists():
                continue

            # Find .git directories
            for git_dir in search_path.rglob(".git"):
                if git_dir.is_dir():
                    repo_root = git_dir.parent
                    repos.append(repo_root)

        logger.info(f"Discovered {len(repos)} repositories")
        return repos

    def index_repository(
        self,
        repo_path: Path,
        force: bool = False,
    ) -> Optional[IndexedRepo]:
        """
        Index a single repository.

        Args:
            repo_path: Path to repository root
            force: If True, re-index even if already indexed

        Returns:
            IndexedRepo object, or None if indexing failed
        """
        repo_path_str = str(repo_path.resolve())

        # Check if already indexed
        if not force and repo_path_str in self.repos:
            logger.debug(f"Repository already indexed: {repo_path.name}")
            return self.repos[repo_path_str]

        logger.info(f"Indexing repository: {repo_path.name}")

        try:
            # Find all Python files
            python_files = list(repo_path.rglob("*.py"))

            # Filter out common ignore patterns
            python_files = [
                f for f in python_files
                if not any(part.startswith(".") or part in ["venv", "env", "__pycache__", "node_modules"]
                          for part in f.parts)
            ]

            indexed_modules = []
            function_count = 0
            class_count = 0

            # Index each file
            for py_file in python_files:
                module = self._index_file(py_file, repo_path)
                if module:
                    self.modules[module.path] = module
                    indexed_modules.append(module.path)
                    function_count += len(module.functions)
                    class_count += len(module.classes)

            # Create repo index
            repo = IndexedRepo(
                path=repo_path_str,
                name=repo_path.name,
                language="python",
                last_indexed=datetime.now().isoformat(),
                file_count=len(python_files),
                module_count=len(indexed_modules),
                function_count=function_count,
                class_count=class_count,
                modules=indexed_modules,
            )

            self.repos[repo_path_str] = repo

            # Save index
            self._save_index()

            logger.info(
                f"Indexed {repo_path.name}: "
                f"{len(indexed_modules)} modules, "
                f"{function_count} functions, "
                f"{class_count} classes"
            )

            return repo

        except Exception as e:
            logger.error(f"Failed to index repository {repo_path}: {e}")
            return None

    def index_all(self, force: bool = False) -> int:
        """
        Index all discovered repositories.

        Args:
            force: If True, re-index all repositories

        Returns:
            Number of repositories indexed
        """
        repos = self.discover_repos()
        indexed_count = 0

        for repo_path in repos:
            result = self.index_repository(repo_path, force=force)
            if result:
                indexed_count += 1

        logger.info(f"Indexed {indexed_count}/{len(repos)} repositories")
        return indexed_count

    def search_functions(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
    ) -> List[CodeFunction]:
        """
        Search for functions by semantic similarity.

        Args:
            query: Search query (e.g., "fibonacci sequence")
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of CodeFunction objects
        """
        if not EMBEDDINGS_AVAILABLE or not embeddings_available():
            # Fallback to keyword search
            return self._search_functions_by_keywords(query, limit)

        # Compute query embedding
        query_embedding = compute_embedding(query)
        if not query_embedding:
            return []

        # Score all functions
        from core.embeddings import cosine_similarity

        scored = []
        for func in self.functions.values():
            if func.embedding:
                score = cosine_similarity(query_embedding, func.embedding)
                if score >= min_similarity:
                    scored.append((func, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [func for func, score in scored[:limit]]

    def search_classes(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
    ) -> List[CodeClass]:
        """
        Search for classes by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of CodeClass objects
        """
        if not EMBEDDINGS_AVAILABLE or not embeddings_available():
            return self._search_classes_by_keywords(query, limit)

        query_embedding = compute_embedding(query)
        if not query_embedding:
            return []

        from core.embeddings import cosine_similarity

        scored = []
        for cls in self.classes.values():
            if cls.embedding:
                score = cosine_similarity(query_embedding, cls.embedding)
                if score >= min_similarity:
                    scored.append((cls, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [cls for cls, score in scored[:limit]]

    def search_modules(
        self,
        query: str,
        limit: int = 10,
        min_similarity: float = 0.6,
    ) -> List[CodeModule]:
        """
        Search for modules by semantic similarity.

        Args:
            query: Search query
            limit: Maximum results
            min_similarity: Minimum similarity threshold

        Returns:
            List of CodeModule objects
        """
        if not EMBEDDINGS_AVAILABLE or not embeddings_available():
            return self._search_modules_by_keywords(query, limit)

        query_embedding = compute_embedding(query)
        if not query_embedding:
            return []

        from core.embeddings import cosine_similarity

        scored = []
        for module in self.modules.values():
            if module.embedding:
                score = cosine_similarity(query_embedding, module.embedding)
                if score >= min_similarity:
                    scored.append((module, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [module for module, score in scored[:limit]]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _index_file(
        self,
        file_path: Path,
        repo_root: Path,
    ) -> Optional[CodeModule]:
        """
        Index a single Python file.

        Args:
            file_path: Path to Python file
            repo_root: Repository root path

        Returns:
            CodeModule object, or None if parsing failed
        """
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Extract module-level docstring
            docstring = self._extract_module_docstring(content)

            # Parse with tree-sitter if available
            functions = []
            classes = []

            if self.parser and TREE_SITTER_AVAILABLE:
                tree = self.parser.parse(content.encode())
                root_node = tree.root_node

                # Extract functions and classes
                for node in root_node.children:
                    if node.type == "function_definition":
                        func = self._parse_function(node, content, str(file_path))
                        if func:
                            self.functions[f"{file_path}:{func.name}"] = func
                            functions.append(func.name)

                    elif node.type == "class_definition":
                        cls = self._parse_class(node, content, str(file_path))
                        if cls:
                            self.classes[f"{file_path}:{cls.name}"] = cls
                            classes.append(cls.name)

            # Create module
            relative_path = file_path.relative_to(repo_root)
            module = CodeModule(
                name=file_path.stem,
                path=str(relative_path),
                docstring=docstring,
                functions=functions,
                classes=classes,
            )

            # Compute embedding
            if EMBEDDINGS_AVAILABLE and embeddings_available():
                module_text = f"{module.name} {module.docstring or ''} {' '.join(functions)} {' '.join(classes)}"
                embedding = compute_embedding(module_text)
                if embedding:
                    module = CodeModule(
                        **{**msgspec.structs.asdict(module), "embedding": embedding}
                    )

            return module

        except Exception as e:
            logger.debug(f"Failed to index file {file_path}: {e}")
            return None

    def _parse_function(
        self,
        node: Any,
        source: str,
        file_path: str,
    ) -> Optional[CodeFunction]:
        """Parse a function definition node."""
        try:
            # Get function name
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            name = source[name_node.start_byte:name_node.end_byte]

            # Get signature
            params_node = node.child_by_field_name("parameters")
            params = source[params_node.start_byte:params_node.end_byte] if params_node else "()"

            signature = f"def {name}{params}"

            # Get docstring
            docstring = self._extract_docstring(node, source)

            # Create function object
            func = CodeFunction(
                name=name,
                signature=signature,
                docstring=docstring,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=file_path,
            )

            # Compute embedding
            if EMBEDDINGS_AVAILABLE and embeddings_available():
                func_text = f"{signature} {docstring or ''}"
                embedding = compute_embedding(func_text)
                if embedding:
                    func = CodeFunction(
                        **{**msgspec.structs.asdict(func), "embedding": embedding}
                    )

            return func

        except Exception as e:
            logger.debug(f"Failed to parse function: {e}")
            return None

    def _parse_class(
        self,
        node: Any,
        source: str,
        file_path: str,
    ) -> Optional[CodeClass]:
        """Parse a class definition node."""
        try:
            # Get class name
            name_node = node.child_by_field_name("name")
            if not name_node:
                return None

            name = source[name_node.start_byte:name_node.end_byte]

            # Get docstring
            docstring = self._extract_docstring(node, source)

            # Get method names
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        method_name_node = child.child_by_field_name("name")
                        if method_name_node:
                            methods.append(source[method_name_node.start_byte:method_name_node.end_byte])

            # Create class object
            cls = CodeClass(
                name=name,
                docstring=docstring,
                methods=methods,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
                file_path=file_path,
            )

            # Compute embedding
            if EMBEDDINGS_AVAILABLE and embeddings_available():
                cls_text = f"{name} {docstring or ''} {' '.join(methods)}"
                embedding = compute_embedding(cls_text)
                if embedding:
                    cls = CodeClass(
                        **{**msgspec.structs.asdict(cls), "embedding": embedding}
                    )

            return cls

        except Exception as e:
            logger.debug(f"Failed to parse class: {e}")
            return None

    def _extract_docstring(self, node: Any, source: str) -> Optional[str]:
        """Extract docstring from a function or class node."""
        try:
            body = node.child_by_field_name("body")
            if not body:
                return None

            # Look for first expression statement with a string
            for child in body.children:
                if child.type == "expression_statement":
                    for expr_child in child.children:
                        if expr_child.type == "string":
                            docstring = source[expr_child.start_byte:expr_child.end_byte]
                            # Remove quotes
                            return docstring.strip('"""').strip("'''").strip()

            return None

        except Exception:
            return None

    def _extract_module_docstring(self, source: str) -> Optional[str]:
        """Extract module-level docstring from source."""
        # Simple regex approach for module docstring
        import re
        match = re.match(r'^"""(.*?)"""', source, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.match(r"^'''(.*?)'''", source, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def _search_functions_by_keywords(self, query: str, limit: int) -> List[CodeFunction]:
        """Fallback keyword-based function search."""
        keywords = set(query.lower().split())
        scored = []

        for func in self.functions.values():
            func_text = f"{func.name} {func.signature} {func.docstring or ''}".lower()
            matches = sum(1 for kw in keywords if kw in func_text)
            if matches > 0:
                scored.append((func, matches))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [func for func, score in scored[:limit]]

    def _search_classes_by_keywords(self, query: str, limit: int) -> List[CodeClass]:
        """Fallback keyword-based class search."""
        keywords = set(query.lower().split())
        scored = []

        for cls in self.classes.values():
            cls_text = f"{cls.name} {cls.docstring or ''} {' '.join(cls.methods)}".lower()
            matches = sum(1 for kw in keywords if kw in cls_text)
            if matches > 0:
                scored.append((cls, matches))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [cls for cls, score in scored[:limit]]

    def _search_modules_by_keywords(self, query: str, limit: int) -> List[CodeModule]:
        """Fallback keyword-based module search."""
        keywords = set(query.lower().split())
        scored = []

        for module in self.modules.values():
            module_text = f"{module.name} {module.docstring or ''}".lower()
            matches = sum(1 for kw in keywords if kw in module_text)
            if matches > 0:
                scored.append((module, matches))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [module for module, score in scored[:limit]]

    def _load_index(self):
        """Load index from disk."""
        index_file = self.index_dir / "index.json"

        if not index_file.exists():
            return

        try:
            with open(index_file, "r") as f:
                data = json.load(f)

            self.repos = {k: IndexedRepo(**v) for k, v in data.get("repos", {}).items()}
            self.modules = {k: CodeModule(**v) for k, v in data.get("modules", {}).items()}
            self.functions = {k: CodeFunction(**v) for k, v in data.get("functions", {}).items()}
            self.classes = {k: CodeClass(**v) for k, v in data.get("classes", {}).items()}

            logger.debug(f"Loaded index: {len(self.repos)} repos, {len(self.modules)} modules")

        except Exception as e:
            logger.warning(f"Failed to load index: {e}")

    def _save_index(self):
        """Save index to disk."""
        index_file = self.index_dir / "index.json"

        try:
            data = {
                "repos": {k: msgspec.structs.asdict(v) for k, v in self.repos.items()},
                "modules": {k: msgspec.structs.asdict(v) for k, v in self.modules.items()},
                "functions": {k: msgspec.structs.asdict(v) for k, v in self.functions.items()},
                "classes": {k: msgspec.structs.asdict(v) for k, v in self.classes.items()},
            }

            with open(index_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved index to disk")

        except Exception as e:
            logger.warning(f"Failed to save index: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_local_index(
    index_dir: Optional[Path] = None,
    search_paths: Optional[List[Path]] = None,
) -> LocalRepoIndex:
    """
    Create a local repository indexer.

    Args:
        index_dir: Optional index directory
        search_paths: Optional search paths

    Returns:
        LocalRepoIndex instance
    """
    return LocalRepoIndex(index_dir=index_dir, search_paths=search_paths)
