"""
PARAGON GITHUB SEARCH - Enhanced GitHub API Integration

Provides robust GitHub API integration with:
- Repository search
- Code search
- File content retrieval
- README extraction for project understanding
- Rate limit handling
- Caching to avoid repeated requests

Design Principles:
1. GRACEFUL DEGRADATION: Works without token (60 req/hour) or with token (5000 req/hour)
2. AGGRESSIVE CACHING: Minimize API calls
3. RATE LIMIT AWARE: Respects GitHub's rate limits
4. MSGSPEC.STRUCT: All data structures
"""

import msgspec
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
import os
import time
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. GitHub search unavailable.")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class GitHubRepo(msgspec.Struct, kw_only=True):
    """Represents a GitHub repository."""
    full_name: str
    html_url: str
    description: Optional[str] = None
    stars: int = 0
    forks: int = 0
    language: Optional[str] = None
    topics: List[str] = []
    last_updated: Optional[str] = None
    readme_content: Optional[str] = None
    license: Optional[str] = None


class GitHubCodeFile(msgspec.Struct, kw_only=True):
    """Represents a code file from GitHub search."""
    name: str
    path: str
    html_url: str
    repository: str
    content: Optional[str] = None
    language: Optional[str] = None


class RateLimitInfo(msgspec.Struct, kw_only=True):
    """GitHub API rate limit information."""
    limit: int
    remaining: int
    reset_time: float  # Unix timestamp


class SearchCacheEntry(msgspec.Struct, kw_only=True):
    """Cache entry for GitHub searches."""
    query: str
    results_json: str
    timestamp: float
    rate_limit: Optional[RateLimitInfo] = None


# =============================================================================
# GITHUB SEARCH CLIENT
# =============================================================================

class GitHubSearch:
    """
    GitHub API client for searching repositories and code.

    Features:
    - Repository search with filtering
    - Code search with language filters
    - README extraction
    - Rate limit tracking
    - Local caching
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize GitHub search client.

        Args:
            token: GitHub personal access token (uses GITHUB_TOKEN env var if not provided)
            cache_dir: Directory for caching results (default: data/github_cache)
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.cache_dir = cache_dir or Path("data/github_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API configuration
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"

        # Rate limiting
        self.rate_limit: Optional[RateLimitInfo] = None
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests

        # Cache settings
        self.cache_ttl = timedelta(hours=24)

        logger.info(
            f"GitHubSearch initialized. "
            f"Token: {'Yes' if self.token else 'No (60 req/hour limit)'}"
        )

    def search_repositories(
        self,
        query: str,
        language: Optional[str] = None,
        sort: str = "stars",
        max_results: int = 10,
        include_readme: bool = False,
    ) -> List[GitHubRepo]:
        """
        Search GitHub repositories.

        Args:
            query: Search query (e.g., "machine learning", "web framework")
            language: Filter by programming language (e.g., "python", "javascript")
            sort: Sort order ("stars", "forks", "updated")
            max_results: Maximum results to return
            include_readme: Whether to fetch README content (slower, more API calls)

        Returns:
            List of GitHubRepo objects
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available")
            return []

        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"

        # Check cache
        cache_key = f"repos_{search_query}_{sort}_{max_results}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Using cached repository search for: {query}")
            return [GitHubRepo(**item) for item in cached]

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("GitHub API rate limit exceeded")
            return []

        try:
            url = f"{self.base_url}/search/repositories"
            params = {
                "q": search_query,
                "sort": sort,
                "order": "desc",
                "per_page": max_results,
            }

            self._wait_for_rate_limit()
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            self.last_request_time = time.time()

            # Update rate limit info
            self._update_rate_limit(response)

            if response.status_code != 200:
                logger.warning(f"GitHub search failed: {response.status_code} - {response.text}")
                return []

            data = response.json()
            repos = []

            for item in data.get("items", [])[:max_results]:
                repo = self._parse_repo(item)

                # Optionally fetch README
                if include_readme:
                    readme = self._fetch_readme(repo.full_name)
                    repo = GitHubRepo(
                        **{**msgspec.structs.asdict(repo), "readme_content": readme}
                    )

                repos.append(repo)

            # Cache results
            self._cache_results(cache_key, repos)

            logger.debug(f"Found {len(repos)} repositories for query: {query}")
            return repos

        except Exception as e:
            logger.error(f"GitHub repository search error: {e}")
            return []

    def search_code(
        self,
        query: str,
        language: Optional[str] = None,
        max_results: int = 10,
        fetch_content: bool = False,
    ) -> List[GitHubCodeFile]:
        """
        Search GitHub code files.

        Args:
            query: Search query (e.g., "def fibonacci", "class GraphDB")
            language: Filter by language (e.g., "python")
            max_results: Maximum results to return
            fetch_content: Whether to fetch file content (slower)

        Returns:
            List of GitHubCodeFile objects
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available")
            return []

        # Build search query
        search_query = query
        if language:
            search_query += f" language:{language}"

        # Check cache
        cache_key = f"code_{search_query}_{max_results}"
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Using cached code search for: {query}")
            return [GitHubCodeFile(**item) for item in cached]

        # Check rate limit
        if not self._check_rate_limit():
            logger.warning("GitHub API rate limit exceeded")
            return []

        try:
            url = f"{self.base_url}/search/code"
            params = {
                "q": search_query,
                "per_page": max_results,
            }

            self._wait_for_rate_limit()
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            self.last_request_time = time.time()

            # Update rate limit
            self._update_rate_limit(response)

            if response.status_code != 200:
                logger.warning(f"GitHub code search failed: {response.status_code}")
                return []

            data = response.json()
            files = []

            for item in data.get("items", [])[:max_results]:
                code_file = self._parse_code_file(item)

                # Optionally fetch content
                if fetch_content and item.get("url"):
                    content = self._fetch_file_content(item["url"])
                    code_file = GitHubCodeFile(
                        **{**msgspec.structs.asdict(code_file), "content": content}
                    )

                files.append(code_file)

            # Cache results
            self._cache_results(cache_key, files)

            logger.debug(f"Found {len(files)} code files for query: {query}")
            return files

        except Exception as e:
            logger.error(f"GitHub code search error: {e}")
            return []

    def _fetch_readme(self, repo_full_name: str) -> Optional[str]:
        """
        Fetch README content for a repository.

        Args:
            repo_full_name: Repository name (e.g., "owner/repo")

        Returns:
            README content as string, or None if not found
        """
        if not REQUESTS_AVAILABLE:
            return None

        try:
            url = f"{self.base_url}/repos/{repo_full_name}/readme"

            self._wait_for_rate_limit()
            response = requests.get(
                url,
                headers={**self.headers, "Accept": "application/vnd.github.v3.raw"},
                timeout=10,
            )
            self.last_request_time = time.time()

            if response.status_code == 200:
                return response.text[:10000]  # Limit to 10KB
            else:
                return None

        except Exception as e:
            logger.debug(f"Failed to fetch README for {repo_full_name}: {e}")
            return None

    def _fetch_file_content(self, file_url: str) -> Optional[str]:
        """
        Fetch file content from GitHub.

        Args:
            file_url: GitHub API URL for the file

        Returns:
            File content as string, or None if failed
        """
        if not REQUESTS_AVAILABLE:
            return None

        try:
            self._wait_for_rate_limit()
            response = requests.get(
                file_url,
                headers={**self.headers, "Accept": "application/vnd.github.v3.raw"},
                timeout=10,
            )
            self.last_request_time = time.time()

            if response.status_code == 200:
                # Try to decode as JSON (for API response)
                try:
                    data = response.json()
                    if "content" in data:
                        import base64
                        return base64.b64decode(data["content"]).decode("utf-8")[:5000]
                except:
                    pass

                # Otherwise return as text
                return response.text[:5000]  # Limit to 5KB
            else:
                return None

        except Exception as e:
            logger.debug(f"Failed to fetch file content: {e}")
            return None

    def _parse_repo(self, item: Dict[str, Any]) -> GitHubRepo:
        """Parse repository data from GitHub API response."""
        return GitHubRepo(
            full_name=item["full_name"],
            html_url=item["html_url"],
            description=item.get("description"),
            stars=item.get("stargazers_count", 0),
            forks=item.get("forks_count", 0),
            language=item.get("language"),
            topics=item.get("topics", []),
            last_updated=item.get("updated_at"),
            license=item.get("license", {}).get("name") if item.get("license") else None,
        )

    def _parse_code_file(self, item: Dict[str, Any]) -> GitHubCodeFile:
        """Parse code file data from GitHub API response."""
        return GitHubCodeFile(
            name=item["name"],
            path=item["path"],
            html_url=item["html_url"],
            repository=item["repository"]["full_name"],
            language=item.get("language"),
        )

    def _check_rate_limit(self) -> bool:
        """
        Check if we have API calls remaining.

        Returns:
            True if we can make requests, False if rate limited
        """
        if self.rate_limit is None:
            return True  # Unknown, assume OK

        if self.rate_limit.remaining > 0:
            return True

        # Check if reset time has passed
        if time.time() >= self.rate_limit.reset_time:
            return True

        return False

    def _update_rate_limit(self, response: Any):
        """Update rate limit information from response headers."""
        try:
            self.rate_limit = RateLimitInfo(
                limit=int(response.headers.get("X-RateLimit-Limit", 60)),
                remaining=int(response.headers.get("X-RateLimit-Remaining", 60)),
                reset_time=float(response.headers.get("X-RateLimit-Reset", time.time() + 3600)),
            )
            logger.debug(
                f"Rate limit: {self.rate_limit.remaining}/{self.rate_limit.limit} remaining"
            )
        except Exception as e:
            logger.debug(f"Failed to update rate limit: {e}")

    def _wait_for_rate_limit(self):
        """Enforce minimum time between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)

    def _get_cached(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve cached results."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                entry_data = json.load(f)

            # Check TTL
            cached_time = datetime.fromtimestamp(entry_data["timestamp"])
            if datetime.now() - cached_time > self.cache_ttl:
                cache_file.unlink()
                return None

            return json.loads(entry_data["results_json"])

        except Exception as e:
            logger.debug(f"Cache read error: {e}")
            return None

    def _cache_results(self, cache_key: str, results: List[Any]):
        """Cache search results."""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            results_dicts = [msgspec.structs.asdict(r) for r in results]

            entry = {
                "timestamp": time.time(),
                "results_json": json.dumps(results_dicts),
            }

            with open(cache_file, "w") as f:
                json.dump(entry, f)

        except Exception as e:
            logger.debug(f"Cache write error: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_github_search(
    token: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> GitHubSearch:
    """
    Create a GitHubSearch client.

    Args:
        token: Optional GitHub token
        cache_dir: Optional cache directory

    Returns:
        GitHubSearch instance
    """
    return GitHubSearch(token=token, cache_dir=cache_dir)
