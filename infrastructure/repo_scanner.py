"""
PARAGON REPOSITORY SCANNER - Multi-Level Code Similarity Detection

Helps identify existing code that could be reused or referenced at three levels:
1. Whole-Project Similarity: "This sounds like repository X"
2. Module/Component Similarity: "This module exists in Y, you could plug it in"
3. Task/Function Similarity: "This specific implementation exists in Z"

Architecture:
- Uses embeddings for semantic similarity (core/embeddings.py)
- Supports GitHub API and local repository indexing
- Caches aggressively to avoid repeated searches
- Integrates with RerunLogger for transparency

Design Principles:
1. MSGSPEC.STRUCT: All data schemas (NO Pydantic)
2. GRACEFUL DEGRADATION: Works without GitHub token or embeddings
3. CACHE-FIRST: Avoid repeated network requests
4. TRANSPARENT LOGGING: All searches logged to RerunLogger
"""

import msgspec
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging
import os
import time
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Try importing optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. GitHub search will be unavailable.")

try:
    from core.embeddings import compute_embedding, cosine_similarity, is_available as embeddings_available
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("Embeddings unavailable. Will use keyword-based similarity only.")

try:
    from infrastructure.rerun_logger import RerunLogger, create_logger
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES (msgspec.Struct)
# =============================================================================

class SimilarProject(msgspec.Struct, kw_only=True):
    """
    A whole project similar to the user's specification.

    Represents Level 1: "This sounds like repository X, do you want to use that instead?"
    """
    name: str
    url: str
    description: str
    similarity_score: float
    key_features: List[str] = []
    tech_stack: List[str] = []
    stars: Optional[int] = None
    last_updated: Optional[str] = None
    language: Optional[str] = None


class SimilarModule(msgspec.Struct, kw_only=True):
    """
    A reusable module/component from another project.

    Represents Level 2: "This module exists in Y, you could plug it in"
    """
    name: str
    source_url: str
    description: str
    similarity_score: float
    integration_complexity: str  # "drop-in", "adapter-needed", "inspiration"
    code_snippet: Optional[str] = None
    file_path: Optional[str] = None
    language: str = "python"
    dependencies: List[str] = []


class SimilarImplementation(msgspec.Struct, kw_only=True):
    """
    A specific code implementation (function/class).

    Represents Level 3: "This specific implementation exists in Z"
    """
    function_name: str
    source_url: str
    description: str
    similarity_score: float
    code: str
    license: str = "unknown"
    language: str = "python"
    tags: List[str] = []


class SearchCacheEntry(msgspec.Struct, kw_only=True):
    """Cache entry for search results."""
    query_hash: str
    results_json: str  # JSON-serialized results
    timestamp: float
    search_level: str  # "project", "module", "implementation"


class RepoMetadata(msgspec.Struct, kw_only=True):
    """Metadata about a local repository."""
    path: str
    name: str
    language: str = "python"
    last_indexed: Optional[str] = None
    file_count: int = 0
    module_count: int = 0


# =============================================================================
# REPOSITORY SCANNER
# =============================================================================

class RepoScanner:
    """
    Scans code repositories for similar projects/modules/tasks.

    Three-level similarity detection:
    1. Project-level: Find repositories solving similar problems
    2. Module-level: Find reusable components
    3. Implementation-level: Find specific code implementations
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
        rerun_logger: Optional[Any] = None,
    ):
        """
        Initialize the repository scanner.

        Args:
            cache_dir: Directory for caching search results (default: data/repo_cache)
            github_token: Optional GitHub API token (uses GITHUB_TOKEN env var if not provided)
            rerun_logger: Optional RerunLogger for visual logging
        """
        self.cache_dir = cache_dir or Path("data/repo_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # GitHub API setup
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.github_headers = {}
        if self.github_token:
            self.github_headers["Authorization"] = f"token {self.github_token}"

        # RerunLogger for transparency
        self.rerun_logger = rerun_logger

        # Cache settings
        self.cache_ttl = timedelta(hours=24)  # Cache for 24 hours

        # Rate limiting
        self.last_github_request = 0
        self.min_request_interval = 1.0  # 1 second between requests

        logger.info(
            f"RepoScanner initialized. "
            f"GitHub token: {'Yes' if self.github_token else 'No'}, "
            f"Embeddings: {'Yes' if EMBEDDINGS_AVAILABLE else 'No'}"
        )

    async def search_similar_projects(
        self,
        description: str,
        sources: List[str] = ["github"],
        max_results: int = 5,
        min_similarity: float = 0.5,
    ) -> List[SimilarProject]:
        """
        Find whole projects similar to description.

        Level 1: "This sounds like repository X, do you want to use that instead?"

        Args:
            description: Project description/specification
            sources: Search sources ("github", "local", etc.)
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of SimilarProject, sorted by similarity descending
        """
        if self.rerun_logger:
            self.rerun_logger.log_thought(
                "repo_scanner",
                f"Searching for similar projects: {description[:100]}..."
            )

        # Check cache first
        cache_key = self._get_cache_key("project", description, sources)
        cached = self._get_cached_results(cache_key)
        if cached:
            logger.debug(f"Using cached project search results for: {description[:50]}")
            return [SimilarProject(**item) for item in cached[:max_results]]

        results = []

        # Search GitHub
        if "github" in sources and REQUESTS_AVAILABLE:
            github_results = self._search_github_repos(description, max_results)
            results.extend(github_results)

        # Search local repositories
        if "local" in sources:
            local_results = self._search_local_repos(description, max_results)
            results.extend(local_results)

        # Compute similarity scores if embeddings available
        if EMBEDDINGS_AVAILABLE and embeddings_available():
            results = self._score_projects_by_embedding(description, results)
        else:
            # Fallback: keyword-based scoring
            results = self._score_projects_by_keywords(description, results)

        # Filter by minimum similarity and sort
        results = [r for r in results if r.similarity_score >= min_similarity]
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Cache results
        self._cache_results(cache_key, results[:max_results])

        if self.rerun_logger:
            self.rerun_logger.log_thought(
                "repo_scanner",
                f"Found {len(results)} similar projects (showing top {max_results})"
            )

        return results[:max_results]

    async def search_similar_modules(
        self,
        module_spec: str,
        language: str = "python",
        max_results: int = 5,
        min_similarity: float = 0.6,
    ) -> List[SimilarModule]:
        """
        Find reusable modules/components.

        Level 2: "This module exists in Y, you could plug it in"

        Args:
            module_spec: Module specification/description
            language: Programming language filter
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of SimilarModule, sorted by similarity descending
        """
        if self.rerun_logger:
            self.rerun_logger.log_thought(
                "repo_scanner",
                f"Searching for similar modules: {module_spec[:100]}..."
            )

        # Check cache
        cache_key = self._get_cache_key("module", f"{module_spec}:{language}", [])
        cached = self._get_cached_results(cache_key)
        if cached:
            logger.debug(f"Using cached module search results")
            return [SimilarModule(**item) for item in cached[:max_results]]

        results = []

        # Search GitHub code
        if REQUESTS_AVAILABLE:
            github_modules = self._search_github_code(module_spec, language, max_results)
            results.extend(github_modules)

        # Search local code index
        local_modules = self._search_local_modules(module_spec, language, max_results)
        results.extend(local_modules)

        # Score by similarity
        if EMBEDDINGS_AVAILABLE and embeddings_available():
            results = self._score_modules_by_embedding(module_spec, results)
        else:
            results = self._score_modules_by_keywords(module_spec, results)

        # Filter and sort
        results = [r for r in results if r.similarity_score >= min_similarity]
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Cache results
        self._cache_results(cache_key, results[:max_results])

        return results[:max_results]

    async def search_similar_implementations(
        self,
        task_description: str,
        context: Optional[str] = None,
        language: str = "python",
        max_results: int = 3,
        min_similarity: float = 0.7,
    ) -> List[SimilarImplementation]:
        """
        Find specific code implementations.

        Level 3: "This specific implementation exists in Z"

        Args:
            task_description: Specific task/function description
            context: Optional context (e.g., class name, module name)
            language: Programming language filter
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold (higher for implementations)

        Returns:
            List of SimilarImplementation, sorted by similarity descending
        """
        if self.rerun_logger:
            self.rerun_logger.log_thought(
                "repo_scanner",
                f"Searching for similar implementations: {task_description[:100]}..."
            )

        # Build full query
        query = task_description
        if context:
            query = f"{context} {task_description}"

        # Check cache
        cache_key = self._get_cache_key("implementation", f"{query}:{language}", [])
        cached = self._get_cached_results(cache_key)
        if cached:
            logger.debug(f"Using cached implementation search results")
            return [SimilarImplementation(**item) for item in cached[:max_results]]

        results = []

        # Search GitHub code with function/class specificity
        if REQUESTS_AVAILABLE:
            github_impls = self._search_github_implementations(
                task_description, context, language, max_results
            )
            results.extend(github_impls)

        # Search local code index
        local_impls = self._search_local_implementations(
            task_description, context, language, max_results
        )
        results.extend(local_impls)

        # Score by similarity
        if EMBEDDINGS_AVAILABLE and embeddings_available():
            results = self._score_implementations_by_embedding(query, results)
        else:
            results = self._score_implementations_by_keywords(query, results)

        # Filter and sort
        results = [r for r in results if r.similarity_score >= min_similarity]
        results.sort(key=lambda x: x.similarity_score, reverse=True)

        # Cache results
        self._cache_results(cache_key, results[:max_results])

        return results[:max_results]

    # =========================================================================
    # GITHUB SEARCH METHODS
    # =========================================================================

    def _search_github_repos(
        self,
        description: str,
        max_results: int = 5,
    ) -> List[SimilarProject]:
        """Search GitHub repositories."""
        if not REQUESTS_AVAILABLE:
            return []

        # Rate limiting
        self._wait_for_rate_limit()

        # Build search query - extract keywords
        keywords = self._extract_keywords(description)
        query = " ".join(keywords[:5])  # Use top 5 keywords

        try:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": max_results,
            }

            response = requests.get(url, headers=self.github_headers, params=params, timeout=10)
            self.last_github_request = time.time()

            if response.status_code != 200:
                logger.warning(f"GitHub search failed: {response.status_code}")
                return []

            data = response.json()
            results = []

            for item in data.get("items", []):
                project = SimilarProject(
                    name=item["full_name"],
                    url=item["html_url"],
                    description=item.get("description", ""),
                    similarity_score=0.0,  # Will be computed later
                    key_features=[],  # Would need README parsing
                    tech_stack=self._extract_tech_stack(item),
                    stars=item.get("stargazers_count"),
                    last_updated=item.get("updated_at"),
                    language=item.get("language"),
                )
                results.append(project)

            logger.debug(f"Found {len(results)} GitHub repositories")
            return results

        except Exception as e:
            logger.warning(f"GitHub search error: {e}")
            return []

    def _search_github_code(
        self,
        module_spec: str,
        language: str = "python",
        max_results: int = 5,
    ) -> List[SimilarModule]:
        """Search GitHub code for modules."""
        if not REQUESTS_AVAILABLE:
            return []

        self._wait_for_rate_limit()

        keywords = self._extract_keywords(module_spec)
        query = f"{' '.join(keywords[:3])} language:{language}"

        try:
            url = "https://api.github.com/search/code"
            params = {
                "q": query,
                "per_page": max_results,
            }

            response = requests.get(url, headers=self.github_headers, params=params, timeout=10)
            self.last_github_request = time.time()

            if response.status_code != 200:
                logger.warning(f"GitHub code search failed: {response.status_code}")
                return []

            data = response.json()
            results = []

            for item in data.get("items", []):
                module = SimilarModule(
                    name=item["name"],
                    source_url=item["html_url"],
                    description=module_spec,  # Would need file parsing for better description
                    similarity_score=0.0,
                    integration_complexity="inspiration",  # Conservative default
                    file_path=item["path"],
                    language=language,
                )
                results.append(module)

            logger.debug(f"Found {len(results)} GitHub code files")
            return results

        except Exception as e:
            logger.warning(f"GitHub code search error: {e}")
            return []

    def _search_github_implementations(
        self,
        task_description: str,
        context: Optional[str],
        language: str = "python",
        max_results: int = 3,
    ) -> List[SimilarImplementation]:
        """Search GitHub for specific implementations."""
        if not REQUESTS_AVAILABLE:
            return []

        self._wait_for_rate_limit()

        # Build query with function/class hints
        keywords = self._extract_keywords(task_description)
        query = f"def {' '.join(keywords[:2])} language:{language}"

        try:
            url = "https://api.github.com/search/code"
            params = {
                "q": query,
                "per_page": max_results,
            }

            response = requests.get(url, headers=self.github_headers, params=params, timeout=10)
            self.last_github_request = time.time()

            if response.status_code != 200:
                return []

            data = response.json()
            results = []

            for item in data.get("items", []):
                impl = SimilarImplementation(
                    function_name=item["name"],
                    source_url=item["html_url"],
                    description=task_description,
                    similarity_score=0.0,
                    code="# Code snippet would require additional API call",
                    language=language,
                )
                results.append(impl)

            return results

        except Exception as e:
            logger.warning(f"GitHub implementation search error: {e}")
            return []

    # =========================================================================
    # LOCAL SEARCH METHODS
    # =========================================================================

    def _search_local_repos(
        self,
        description: str,
        max_results: int = 5,
    ) -> List[SimilarProject]:
        """Search local repositories (placeholder - would need indexing)."""
        # TODO: Implement local repository indexing
        # This would scan ~/code, ~/projects, etc. and build an index
        return []

    def _search_local_modules(
        self,
        module_spec: str,
        language: str = "python",
        max_results: int = 5,
    ) -> List[SimilarModule]:
        """Search local code for modules (placeholder)."""
        # TODO: Implement local module indexing
        # This would use tree-sitter or AST parsing to find modules
        return []

    def _search_local_implementations(
        self,
        task_description: str,
        context: Optional[str],
        language: str = "python",
        max_results: int = 3,
    ) -> List[SimilarImplementation]:
        """Search local code for implementations (placeholder)."""
        # TODO: Implement local implementation search
        # This would use AST parsing to find matching functions/classes
        return []

    # =========================================================================
    # SIMILARITY SCORING
    # =========================================================================

    def _score_projects_by_embedding(
        self,
        query: str,
        projects: List[SimilarProject],
    ) -> List[SimilarProject]:
        """Score projects using embedding similarity."""
        query_embedding = compute_embedding(query)
        if not query_embedding:
            return self._score_projects_by_keywords(query, projects)

        scored = []
        for project in projects:
            # Create text representation of project
            project_text = f"{project.name} {project.description}"
            if project.key_features:
                project_text += " " + " ".join(project.key_features)

            project_embedding = compute_embedding(project_text)
            if project_embedding:
                score = cosine_similarity(query_embedding, project_embedding)
                # Boost score for popular/recently updated projects
                if project.stars and project.stars > 100:
                    score = min(1.0, score * 1.1)
            else:
                score = 0.3  # Default low score

            scored.append(SimilarProject(
                **{**msgspec.structs.asdict(project), "similarity_score": score}
            ))

        return scored

    def _score_modules_by_embedding(
        self,
        query: str,
        modules: List[SimilarModule],
    ) -> List[SimilarModule]:
        """Score modules using embedding similarity."""
        query_embedding = compute_embedding(query)
        if not query_embedding:
            return self._score_modules_by_keywords(query, modules)

        scored = []
        for module in modules:
            module_text = f"{module.name} {module.description}"
            if module.code_snippet:
                module_text += " " + module.code_snippet[:500]

            module_embedding = compute_embedding(module_text)
            if module_embedding:
                score = cosine_similarity(query_embedding, module_embedding)
            else:
                score = 0.3

            scored.append(SimilarModule(
                **{**msgspec.structs.asdict(module), "similarity_score": score}
            ))

        return scored

    def _score_implementations_by_embedding(
        self,
        query: str,
        implementations: List[SimilarImplementation],
    ) -> List[SimilarImplementation]:
        """Score implementations using embedding similarity."""
        query_embedding = compute_embedding(query)
        if not query_embedding:
            return self._score_implementations_by_keywords(query, implementations)

        scored = []
        for impl in implementations:
            impl_text = f"{impl.function_name} {impl.description} {impl.code[:500]}"

            impl_embedding = compute_embedding(impl_text)
            if impl_embedding:
                score = cosine_similarity(query_embedding, impl_embedding)
            else:
                score = 0.3

            scored.append(SimilarImplementation(
                **{**msgspec.structs.asdict(impl), "similarity_score": score}
            ))

        return scored

    def _score_projects_by_keywords(
        self,
        query: str,
        projects: List[SimilarProject],
    ) -> List[SimilarProject]:
        """Fallback: score by keyword overlap."""
        query_keywords = set(self._extract_keywords(query))

        scored = []
        for project in projects:
            project_text = f"{project.name} {project.description}".lower()
            project_keywords = set(self._extract_keywords(project_text))

            if not query_keywords or not project_keywords:
                score = 0.3
            else:
                overlap = len(query_keywords & project_keywords)
                score = overlap / max(len(query_keywords), len(project_keywords))

            scored.append(SimilarProject(
                **{**msgspec.structs.asdict(project), "similarity_score": score}
            ))

        return scored

    def _score_modules_by_keywords(
        self,
        query: str,
        modules: List[SimilarModule],
    ) -> List[SimilarModule]:
        """Fallback: score modules by keyword overlap."""
        query_keywords = set(self._extract_keywords(query))

        scored = []
        for module in modules:
            module_text = f"{module.name} {module.description}".lower()
            module_keywords = set(self._extract_keywords(module_text))

            overlap = len(query_keywords & module_keywords)
            score = overlap / max(len(query_keywords), len(module_keywords)) if query_keywords else 0.3

            scored.append(SimilarModule(
                **{**msgspec.structs.asdict(module), "similarity_score": score}
            ))

        return scored

    def _score_implementations_by_keywords(
        self,
        query: str,
        implementations: List[SimilarImplementation],
    ) -> List[SimilarImplementation]:
        """Fallback: score implementations by keyword overlap."""
        query_keywords = set(self._extract_keywords(query))

        scored = []
        for impl in implementations:
            impl_text = f"{impl.function_name} {impl.description} {impl.code}".lower()
            impl_keywords = set(self._extract_keywords(impl_text))

            overlap = len(query_keywords & impl_keywords)
            score = overlap / max(len(query_keywords), len(impl_keywords)) if query_keywords else 0.3

            scored.append(SimilarImplementation(
                **{**msgspec.structs.asdict(impl), "similarity_score": score}
            ))

        return scored

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "be", "are", "this",
            "that", "it", "i", "you", "we", "they", "he", "she"
        }

        # Simple tokenization and filtering
        words = text.lower().split()
        keywords = [w.strip(".,!?;:") for w in words if len(w) > 2 and w not in stop_words]

        return keywords

    def _extract_tech_stack(self, repo_item: Dict[str, Any]) -> List[str]:
        """Extract technology stack from GitHub repo metadata."""
        stack = []

        if repo_item.get("language"):
            stack.append(repo_item["language"])

        # Would need topics/tags from GitHub API for more details
        if "topics" in repo_item:
            stack.extend(repo_item["topics"][:5])

        return stack

    def _wait_for_rate_limit(self):
        """Enforce rate limiting for GitHub API."""
        elapsed = time.time() - self.last_github_request
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _get_cache_key(self, level: str, query: str, sources: List[str]) -> str:
        """Generate cache key for a search."""
        key_parts = [level, query, ",".join(sorted(sources))]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _get_cached_results(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached search results."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                entry_data = json.load(f)

            # Check TTL
            cached_time = datetime.fromtimestamp(entry_data["timestamp"])
            if datetime.now() - cached_time > self.cache_ttl:
                logger.debug(f"Cache expired for {cache_key}")
                cache_file.unlink()  # Remove expired cache
                return None

            return json.loads(entry_data["results_json"])

        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def _cache_results(self, cache_key: str, results: List[Any]):
        """Cache search results."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            # Convert msgspec structs to dicts
            results_dicts = [msgspec.structs.asdict(r) for r in results]

            entry = {
                "timestamp": time.time(),
                "results_json": json.dumps(results_dicts),
            }

            with open(cache_file, "w") as f:
                json.dump(entry, f)

            logger.debug(f"Cached {len(results)} results with key {cache_key[:16]}...")

        except Exception as e:
            logger.debug(f"Cache write error: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_scanner(
    cache_dir: Optional[Path] = None,
    github_token: Optional[str] = None,
    rerun_logger: Optional[Any] = None,
) -> RepoScanner:
    """
    Factory function to create a RepoScanner instance.

    Args:
        cache_dir: Optional cache directory
        github_token: Optional GitHub token
        rerun_logger: Optional RerunLogger for transparency

    Returns:
        RepoScanner instance
    """
    return RepoScanner(
        cache_dir=cache_dir,
        github_token=github_token,
        rerun_logger=rerun_logger,
    )
