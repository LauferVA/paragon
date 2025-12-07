# Paragon Similarity Search System

## Overview

The Paragon Similarity Search System helps identify existing code that could be reused or referenced, operating at three distinct levels:

### Level 1: Whole-Project Similarity
**"This sounds like repository X, do you want to use that instead?"**

Prevents building something that already exists by finding similar projects on GitHub.

**Example:**
- User wants to build a "graph database for code dependencies"
- System finds: neo4j, rustworkx, networkx
- Suggestion: "Consider using rustworkx instead of building from scratch"

### Level 2: Module/Component Similarity
**"This module exists in Y, you could plug it in"**

Identifies reusable components that could be integrated into the project.

**Example:**
- User needs: "tree-sitter parser for Python"
- System finds: tree-sitter-python library
- Suggestion: "Drop-in candidate with minimal changes"

### Level 3: Task/Function Similarity
**"This specific implementation exists in Z"**

Locates specific code implementations for algorithms or patterns.

**Example:**
- User needs: "topological sort for DAG"
- System finds: Implementation in networkx
- Suggestion: "Exact match - review this implementation"

---

## Architecture

### Components

```
infrastructure/
├── repo_scanner.py          # Main similarity search engine
├── github_search.py         # GitHub API integration
└── local_repo_index.py      # Local repository indexing

agents/
├── similarity_advisor.py    # User-facing suggestions
└── research_similarity.py   # Integration with research phase
```

### Data Flow

```
User Spec
    ↓
research_node (orchestrator)
    ↓
search_similar_code_sync()
    ↓
RepoScanner
    ├── GitHub Search (projects, code)
    ├── Local Index (if available)
    └── Embedding-based scoring
    ↓
SimilarityAdvisor
    ├── Format suggestions
    └── Determine thresholds
    ↓
RESEARCH node (graph)
    └── Linked to REQ node
```

---

## Usage

### Basic Usage

```python
from infrastructure.repo_scanner import create_scanner
from agents.similarity_advisor import create_advisor

# Create scanner
scanner = create_scanner()
advisor = create_advisor()

# Search for similar projects
projects = await scanner.search_similar_projects(
    description="Build a REST API with authentication",
    max_results=5,
    min_similarity=0.5,
)

# Format suggestions
for project in projects:
    print(advisor.format_project_suggestion(project, ""))
```

### Integration with Orchestrator

The similarity search is automatically integrated into the research phase:

```python
# In research_node()
from agents.research_similarity import search_similar_code_sync

# Run similarity search
similarity_results = search_similar_code_sync(
    spec=spec,
    session_id=session_id,
    req_node_id=req_node_id,
    rerun_logger=rerun_logger,
    enable_project_search=True,
    enable_module_search=True,
    enable_implementation_search=False,  # More expensive
)

# Results are stored in RESEARCH node
# suggestions = similarity_results["suggestions"]
# research_node_id = similarity_results["research_node_id"]
```

### Configuration

Set environment variables to configure behavior:

```bash
# Enable/disable similarity search
export PARAGON_SIMILARITY_SEARCH=true

# Enable specific levels
export PARAGON_SIMILARITY_PROJECTS=true
export PARAGON_SIMILARITY_MODULES=true
export PARAGON_SIMILARITY_IMPLS=false  # More expensive, off by default

# GitHub token for higher rate limits (5000/hour vs 60/hour)
export GITHUB_TOKEN=your_github_token_here
```

---

## Similarity Thresholds

### Projects (Level 1)
- **0.7+**: Strong match → "Use this instead of building from scratch"
- **0.5-0.7**: Possible alternative → "Review for ideas or components"
- **< 0.5**: Weak match → "Listed for reference only"

### Modules (Level 2)
- **0.7+**: Drop-in candidate → "Works with minimal changes"
- **0.6-0.7**: Adapter needed → "Wrap with adapter layer"
- **< 0.6**: Inspiration only → "Review approach but expect rewrite"

### Implementations (Level 3)
- **0.8+**: Exact match → "Solves your exact problem"
- **0.7-0.8**: Similar approach → "Review for pattern/algorithm"
- **< 0.7**: Related → "Might provide useful context"

---

## Similarity Scoring

### With Embeddings (Preferred)

When `sentence-transformers` is available:

1. Compute embedding for query text
2. Compute embeddings for candidates
3. Calculate cosine similarity
4. Boost scores for popular/recent projects

```python
from core.embeddings import compute_embedding, cosine_similarity

query_embedding = compute_embedding(query)
candidate_embedding = compute_embedding(candidate_text)
score = cosine_similarity(query_embedding, candidate_embedding)
```

### Without Embeddings (Fallback)

Keyword-based scoring:

1. Extract keywords from query
2. Extract keywords from candidate
3. Calculate overlap ratio
4. Score = overlap / max(query_keywords, candidate_keywords)

---

## Caching

All search results are cached for 24 hours to minimize API calls:

- **Cache location**: `data/repo_cache/`
- **Cache TTL**: 24 hours
- **Cache key**: Hash of (level, query, sources)

To clear cache:

```bash
rm -rf data/repo_cache/
```

---

## GitHub API Integration

### Rate Limits

- **Without token**: 60 requests/hour
- **With token**: 5000 requests/hour

### Setting Up GitHub Token

1. Go to GitHub Settings → Developer Settings → Personal Access Tokens
2. Generate new token (classic)
3. No special scopes needed for public repos
4. Set environment variable:

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### API Endpoints Used

- **Repository Search**: `/search/repositories`
- **Code Search**: `/search/code`
- **README Fetch**: `/repos/{owner}/{repo}/readme`
- **File Content**: API URL from search results

---

## Local Repository Indexing

### Setup

Index local repositories for faster, offline search:

```python
from infrastructure.local_repo_index import create_local_index

# Create index
index = create_local_index()

# Discover and index all repos
index.index_all()

# Search indexed code
functions = index.search_functions("topological sort", limit=10)
modules = index.search_modules("graph database", limit=10)
classes = index.search_classes("parser", limit=10)
```

### Index Storage

- **Location**: `data/local_index/`
- **Format**: JSON (msgspec-serialized)
- **Contents**: Repos, modules, functions, classes, embeddings

### Incremental Updates

The index supports incremental updates:

```python
# Re-index only changed files
index.index_repository(repo_path, force=False)

# Force full re-index
index.index_repository(repo_path, force=True)
```

---

## Data Structures

All data structures use `msgspec.Struct` (NO Pydantic):

### SimilarProject

```python
class SimilarProject(msgspec.Struct, kw_only=True):
    name: str
    url: str
    description: str
    similarity_score: float
    key_features: List[str] = []
    tech_stack: List[str] = []
    stars: Optional[int] = None
    last_updated: Optional[str] = None
    language: Optional[str] = None
```

### SimilarModule

```python
class SimilarModule(msgspec.Struct, kw_only=True):
    name: str
    source_url: str
    description: str
    similarity_score: float
    integration_complexity: str  # "drop-in", "adapter-needed", "inspiration"
    code_snippet: Optional[str] = None
    file_path: Optional[str] = None
    language: str = "python"
    dependencies: List[str] = []
```

### SimilarImplementation

```python
class SimilarImplementation(msgspec.Struct, kw_only=True):
    function_name: str
    source_url: str
    description: str
    similarity_score: float
    code: str
    license: str = "unknown"
    language: str = "python"
    tags: List[str] = []
```

---

## RerunLogger Integration

All searches are logged to RerunLogger for transparency:

```python
# Logged events:
# - "Searching for similar projects..."
# - "Found {N} similar projects"
# - "Similarity search complete"

if rerun_logger:
    rerun_logger.log_thought(
        "repo_scanner",
        f"Searching for similar projects: {description[:100]}..."
    )
```

View in Rerun viewer:

```bash
rerun data/sessions/*.rrd
```

---

## Performance Considerations

### GitHub API Calls

- **Projects**: 1 API call per search
- **Modules**: 1 API call per search
- **Implementations**: 1 API call + optional content fetches
- **README**: 1 API call per project (if enabled)

**Optimization**: Cache aggressively, batch requests where possible

### Embedding Computation

- **Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Speed**: ~4000 sentences/sec on CPU
- **Memory**: ~22M parameters, ~85MB

**Optimization**: Batch compute embeddings, cache results

### Local Indexing

- **Initial Index**: ~1-5 seconds per repository
- **Search**: O(n) scan, ~100ms for 1000 functions
- **Disk**: ~1MB per 100 repositories

**Optimization**: Use incremental indexing, consider approximate nearest neighbors for large indexes

---

## Examples

### Example 1: Find Similar Projects

```python
import asyncio
from infrastructure.repo_scanner import create_scanner

async def find_projects():
    scanner = create_scanner()

    projects = await scanner.search_similar_projects(
        description="Build a web scraper with rate limiting",
        sources=["github"],
        max_results=5,
    )

    for project in projects:
        print(f"{project.name}: {project.similarity_score:.0%}")
        print(f"  {project.url}")

asyncio.run(find_projects())
```

### Example 2: Find Reusable Modules

```python
import asyncio
from infrastructure.repo_scanner import create_scanner

async def find_modules():
    scanner = create_scanner()

    modules = await scanner.search_similar_modules(
        module_spec="rate limiting middleware for async requests",
        language="python",
        max_results=5,
    )

    for module in modules:
        print(f"{module.name}: {module.integration_complexity}")
        print(f"  {module.source_url}")

asyncio.run(find_modules())
```

### Example 3: Integration with Research

```python
from agents.research_similarity import search_similar_code_sync

# Called from research_node in orchestrator
results = search_similar_code_sync(
    spec="Build a graph database",
    session_id="test_session",
    enable_project_search=True,
    enable_module_search=True,
)

print(results["suggestions"])  # User-facing formatted output
# RESEARCH node created with ID: results["research_node_id"]
```

---

## Troubleshooting

### No Results Found

**Possible causes:**
- Query too vague or specific
- Similarity threshold too high
- GitHub API rate limit exceeded
- No internet connection

**Solutions:**
- Broaden query with more keywords
- Lower `min_similarity` threshold
- Set `GITHUB_TOKEN` for higher limits
- Check cache for previous results

### Slow Performance

**Possible causes:**
- Fetching README content (slow)
- Computing embeddings for large results
- Network latency

**Solutions:**
- Set `include_readme=False`
- Reduce `max_results`
- Use cached results
- Index local repositories

### Import Errors

**Possible causes:**
- Missing dependencies (requests, sentence-transformers)
- Tree-sitter not installed

**Solutions:**

```bash
pip install requests
pip install sentence-transformers
pip install tree-sitter tree-sitter-python
```

---

## Future Enhancements

### Planned Features

1. **Vector Database Integration**: Use FAISS/Pinecone for faster similarity search
2. **Cross-Language Support**: Index JavaScript, TypeScript, Rust, etc.
3. **Dependency Analysis**: Extract dependencies from package.json, requirements.txt
4. **License Compatibility**: Check license compatibility before suggesting
5. **Local Git History**: Search user's own commit history
6. **Collaborative Filtering**: Learn from which suggestions users accept

### Integration Points

- **Architect Agent**: Suggest similar architectures before planning
- **Builder Agent**: Suggest implementations during code generation
- **Tester Agent**: Find similar test strategies
- **Quality Gate**: Compare against similar code quality metrics

---

## References

- GitHub Search API: https://docs.github.com/en/rest/search
- Sentence Transformers: https://www.sbert.net/
- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- Paragon Embeddings: `/Users/lauferva/paragon/core/embeddings.py`
- Paragon Research Phase: `/Users/lauferva/paragon/agents/orchestrator.py#research_node`
