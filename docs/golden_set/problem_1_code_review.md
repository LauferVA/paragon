# GOLDEN SET PROBLEM 1: Multi-Agent Code Review System

**Complexity Level:** High
**Category:** Multi-Agent Systems, External Integration, Real-time Processing
**Estimated Components:** 15-20
**Test Coverage Target:** 100% (production quality gate)

---

## EXECUTIVE SUMMARY

Build an automated code review system that analyzes pull requests from GitHub, detects code smells using multiple analysis strategies, suggests improvements with natural language explanations, tracks review comments as graph nodes, and learns from human reviewer feedback to improve future suggestions.

This problem tests:
- Multi-agent coordination (Analyzer, Suggester, Learner agents)
- External API integration (GitHub API, tree-sitter parsing)
- Graph-native state management (review sessions as subgraphs)
- Real-time streaming updates (webhooks, SSE)
- Quality metrics and learning loops
- Complex edge case handling

---

## 1. FUNCTIONAL REQUIREMENTS

### FR-1: GitHub Integration
**Priority:** P0 (Blocking)

The system MUST integrate with GitHub's REST API and webhooks to:
- Authenticate via GitHub App installation or OAuth token
- Listen for pull request events (opened, synchronized, review_requested)
- Fetch PR diff, changed files, and existing review comments
- Post review comments back to GitHub with inline code suggestions
- Update comment threads with follow-up analysis

**Input Contract:**
```python
class PullRequestEvent(msgspec.Struct):
    repo_owner: str
    repo_name: str
    pr_number: int
    action: Literal["opened", "synchronized", "review_requested"]
    head_sha: str
    changed_files: List[ChangedFile]

class ChangedFile(msgspec.Struct):
    filename: str
    status: Literal["added", "modified", "deleted"]
    patch: str  # Unified diff format
    additions: int
    deletions: int
```

**Output Contract:**
```python
class ReviewComment(msgspec.Struct):
    path: str
    line: int
    body: str  # Markdown-formatted suggestion
    severity: Literal["error", "warning", "info", "suggestion"]
    suggestion_code: Optional[str]  # GitHub suggestion syntax
```

**Edge Cases:**
1. PR with 100+ changed files (batch processing)
2. Binary files in diff (skip with warning)
3. Rate limit exceeded (exponential backoff, queue)
4. Merge conflicts in PR (detect and report)
5. Force-push during review (invalidate in-progress analysis)

**Test Scenarios:**
- Mock GitHub API responses for all event types
- Simulate rate limit errors and verify retry logic
- Test concurrent PR analysis (multiple PRs simultaneously)
- Verify idempotency (same PR analyzed twice yields same results)

---

### FR-2: Static Analysis Engine
**Priority:** P0 (Blocking)

The system MUST perform multi-layered static analysis:
- **Syntax Analysis:** Parse code with tree-sitter to detect syntax errors
- **Linting:** Run ruff/pylint for style violations
- **Type Checking:** Use mypy for type safety issues
- **Security Scanning:** Detect OWASP Top 10 patterns (SQL injection, XSS, hardcoded secrets)
- **Complexity Metrics:** Calculate cyclomatic complexity, cognitive complexity, maintainability index
- **Code Duplication:** Identify clones using abstract syntax tree similarity

**Input Contract:**
```python
class CodeFile(msgspec.Struct):
    path: str
    language: str  # "python", "typescript", "rust", etc.
    content: str
    is_test: bool
```

**Output Contract:**
```python
class AnalysisResult(msgspec.Struct):
    file_path: str
    issues: List[CodeIssue]
    metrics: CodeMetrics

class CodeIssue(msgspec.Struct):
    rule_id: str  # "E501", "SEC101", "COMPLEXITY"
    severity: Literal["error", "warning", "info"]
    line: int
    column: int
    message: str
    suggestion: Optional[str]

class CodeMetrics(msgspec.Struct):
    cyclomatic_complexity: int
    cognitive_complexity: int
    maintainability_index: float
    duplication_percentage: float
    test_coverage_estimate: float
```

**Edge Cases:**
1. File with no extension (use heuristics for language detection)
2. Mixed-language files (Jupyter notebooks, Vue SFCs)
3. Generated code (machine-generated, minified) - skip analysis
4. Very large files (>10k LOC) - sample or timeout
5. Encoding issues (non-UTF-8) - detect and handle

**Test Scenarios:**
- Fixture files for each language with known issues
- Property-based testing: Any valid Python code should parse
- Security test suite: OWASP test cases should trigger warnings
- Performance: Analyze 1000-line file in <500ms (Protocol Alpha)

---

### FR-3: Semantic Code Smell Detection
**Priority:** P1 (High)

The system MUST detect semantic anti-patterns beyond syntax:
- **God Class:** Class with >20 methods or >500 LOC
- **Long Method:** Function with >50 LOC or cyclomatic complexity >15
- **Feature Envy:** Method accessing more external data than internal
- **Data Clumps:** Same 3+ parameters repeated across multiple functions
- **Shotgun Surgery:** Single change requires editing 5+ files
- **Inappropriate Intimacy:** Class accessing private members of another
- **Dead Code:** Unreachable code paths, unused imports

**Algorithm:**
Use graph-based analysis:
1. Build Code Property Graph (CPG) with nodes: CLASS, FUNCTION, CALL, VARIABLE
2. Add edges: CONTAINS, REFERENCES, CALLS, ACCESSES
3. Run graph queries for smell patterns
4. Score by severity (frequency × impact)

**Input Contract:**
```python
class CodePropertyGraph(msgspec.Struct):
    nodes: List[CPGNode]
    edges: List[CPGEdge]

class CPGNode(msgspec.Struct):
    id: str
    type: Literal["class", "function", "variable", "call"]
    name: str
    file_path: str
    line_start: int
    line_end: int
    metadata: Dict[str, Any]  # e.g., {"param_count": 5}
```

**Output Contract:**
```python
class CodeSmell(msgspec.Struct):
    smell_type: str  # "god_class", "long_method", etc.
    confidence: float  # 0.0 to 1.0
    location: CodeLocation
    explanation: str  # Natural language explanation
    refactoring_suggestion: str
    impact_score: float  # Weighted by codebase context
```

**Edge Cases:**
1. Framework code (Django models with many auto-generated methods) - whitelist
2. Test files (long test functions acceptable) - relax thresholds
3. DSLs (SQL builders, config DSLs) - pattern-specific rules
4. Incremental changes (smell exists pre-PR) - only flag new smells

**Test Scenarios:**
- Fixture codebase with known smells (one per type)
- Refactored version should have zero smells
- Test on real open-source repos (scikit-learn, Django)
- Compare with SonarQube/CodeClimate results

---

### FR-4: LLM-Powered Suggestion Generator
**Priority:** P1 (High)

The system MUST use LLM to generate human-readable suggestions:
- Context: Provide surrounding code (±10 lines) for each issue
- Explanation: Natural language description of the problem
- Suggestion: Concrete code fix with diff highlighting
- Rationale: Why this matters (performance, security, maintainability)

**Prompt Engineering:**
```
You are a code reviewer. Analyze this code snippet:

{code_context}

Issue detected: {issue_type} at line {line}
Violation: {rule_message}

Provide:
1. Explanation (1-2 sentences, assume reader is experienced)
2. Suggested fix (show exact code change)
3. Rationale (why this improves the code)

Format as GitHub review comment with suggestion syntax.
```

**Input Contract:**
```python
class SuggestionRequest(msgspec.Struct):
    file_path: str
    issue: CodeIssue
    context_lines_before: List[str]
    context_lines_after: List[str]
    language: str
```

**Output Contract:**
```python
class LLMSuggestion(msgspec.Struct):
    explanation: str
    suggested_code: str
    rationale: str
    confidence: float  # LLM self-assessment
    alternative_approaches: List[str]  # Optional alternatives
```

**Edge Cases:**
1. LLM refuses (unsafe code, ambiguous fix) - fallback to generic message
2. Suggested code is syntactically invalid - validate before posting
3. Suggestion conflicts with existing code - detect and warn
4. Rate limit (API quota) - queue and batch
5. Timeout (slow LLM) - use cached response or skip

**Test Scenarios:**
- Mock LLM responses for deterministic testing
- Validate all suggestions are syntactically correct
- Human eval: Show suggestions to real developers, measure helpfulness
- A/B test: LLM suggestions vs. rule-based messages

---

### FR-5: Review Session Graph Management
**Priority:** P0 (Blocking)

The system MUST model each PR review as a subgraph:
- Create SESSION node for each PR
- Create ANALYSIS nodes for each file analyzed
- Create ISSUE nodes for each detected problem
- Create SUGGESTION nodes for LLM-generated fixes
- Link via edges: SESSION → ANALYSIS → ISSUE → SUGGESTION
- Track status: PENDING → PROCESSING → REVIEWED → POSTED

**Graph Structure:**
```
SESSION (pr_12345)
  ├─ ANALYSIS (file: api/views.py)
  │    ├─ ISSUE (E501: line too long)
  │    │    └─ SUGGESTION (break into multiple lines)
  │    └─ ISSUE (SEC101: SQL injection risk)
  │         └─ SUGGESTION (use parameterized query)
  └─ ANALYSIS (file: api/models.py)
       └─ ISSUE (COMPLEXITY: cyclomatic > 15)
            └─ SUGGESTION (extract helper function)
```

**Node Types:**
```python
class SessionNode(msgspec.Struct):
    type: Literal["SESSION"]
    pr_url: str
    repo: str
    status: Literal["pending", "analyzing", "reviewed", "posted"]
    created_at: str

class AnalysisNode(msgspec.Struct):
    type: Literal["ANALYSIS"]
    file_path: str
    language: str
    analysis_duration_ms: int

class IssueNode(msgspec.Struct):
    type: Literal["ISSUE"]
    rule_id: str
    severity: str
    line: int
    message: str

class SuggestionNode(msgspec.Struct):
    type: Literal["SUGGESTION"]
    explanation: str
    suggested_code: str
    posted_to_github: bool
```

**Edge Cases:**
1. Duplicate analysis (same file analyzed twice) - deduplicate by hash
2. Session abandonment (PR closed mid-review) - cleanup orphaned nodes
3. Concurrent sessions (multiple PRs) - isolate via session ID
4. Graph size explosion (100+ files) - pagination, lazy loading

**Test Scenarios:**
- Create session graph for sample PR, verify structure
- Test teleology: All SUGGESTION nodes trace back to SESSION
- Test cleanup: Delete session, verify all child nodes removed
- Performance: Create 1000-node session in <1s

---

### FR-6: Comment Tracking and Threading
**Priority:** P1 (High)

The system MUST track review comment lifecycle:
- Map graph SUGGESTION nodes to GitHub comment IDs
- Detect when human reviewer replies to bot comment
- Update SUGGESTION status based on feedback (accepted, rejected, modified)
- Create feedback edges: HUMAN_REPLY → SUGGESTION

**Workflow:**
1. Bot posts SUGGESTION as GitHub comment → get comment_id
2. Store mapping: SUGGESTION.data["github_comment_id"] = comment_id
3. Webhook receives comment_reply event → match comment_id
4. Create HUMAN_REPLY node with content
5. Add edge: HUMAN_REPLY → SUGGESTION (FEEDBACK_FOR)
6. Update SUGGESTION.status = "accepted" | "rejected"

**Input Contract:**
```python
class CommentReplyEvent(msgspec.Struct):
    pr_number: int
    comment_id: str  # Original comment being replied to
    reply_id: str
    author: str
    body: str  # Reply text
    created_at: str
```

**Output Contract:**
```python
class FeedbackClassification(msgspec.Struct):
    sentiment: Literal["positive", "negative", "neutral"]
    action: Literal["accepted", "rejected", "needs_clarification"]
    extracted_intent: str
```

**Edge Cases:**
1. Reply is ambiguous ("hmm, maybe") - classify as neutral
2. Reply contains counter-suggestion - create new SUGGESTION node
3. Multiple replies in thread - chain as linked list
4. Outdated comment (line changed) - mark as stale

**Test Scenarios:**
- Post comment, simulate reply, verify graph update
- Test sentiment classification on real GitHub comment corpus
- Verify thread reconstruction from graph traversal

---

### FR-7: Learning from Human Feedback
**Priority:** P2 (Medium)

The system MUST learn from reviewer feedback:
- Track which suggestions are accepted vs. rejected
- Identify patterns: "Rule X suggestions are always rejected for test files"
- Adjust confidence scoring: Lower confidence for frequently rejected patterns
- Store learning in graph: LEARNING_PATTERN nodes

**Learning Strategy:**
```python
class LearningPattern(msgspec.Struct):
    pattern_id: str
    rule_id: str
    context_filter: Dict[str, Any]  # {"file_type": "test", "language": "python"}
    acceptance_rate: float
    sample_size: int
    confidence_adjustment: float  # -0.2 means reduce confidence by 20%
```

**Algorithm:**
1. Aggregate feedback: Group by (rule_id, file_type, language)
2. Calculate acceptance_rate = accepted / (accepted + rejected)
3. If acceptance_rate < 0.3 and sample_size > 10: Create suppression pattern
4. If acceptance_rate > 0.8 and sample_size > 10: Boost confidence
5. Apply pattern: When generating suggestion, check patterns and adjust

**Edge Cases:**
1. Insufficient data (sample_size < 10) - use global baseline
2. Conflicting patterns (different repos have different norms) - repo-specific learning
3. Drift over time (codebase evolves) - decay old patterns
4. Adversarial feedback (troll accepts everything) - detect outliers

**Test Scenarios:**
- Simulate feedback loop: 100 suggestions, 20% accepted
- Verify pattern created after threshold
- Verify future suggestions adjusted
- Test decay: Old patterns should expire

---

### FR-8: Performance and Scalability
**Priority:** P0 (Blocking)

The system MUST meet performance targets:
- **Protocol Alpha:** Analyze single file in <500ms
- **Throughput:** Handle 10 concurrent PR reviews
- **Latency:** Post first comment within 30s of PR creation
- **Memory:** Stay under 1GB RAM for 100-file PR
- **Cost:** <$0.10 per PR review (LLM costs)

**Optimization Strategies:**
1. Incremental analysis: Only analyze changed lines, not entire file
2. Caching: Hash file content, reuse previous analysis if unchanged
3. Parallel processing: Analyze files concurrently (max 5 workers)
4. Model routing: Use fast/cheap model for simple issues, expensive model for complex
5. Batching: Queue GitHub API calls, batch POST requests

**Resource Guard:**
```python
class ReviewResourceGuard:
    def check_before_analysis(self, pr_event):
        if len(pr_event.changed_files) > 100:
            return ResourceSignal.WARNING  # Batch process
        if self.active_sessions > 10:
            return ResourceSignal.CRITICAL  # Queue
        return ResourceSignal.OK
```

**Edge Cases:**
1. Massive PR (1000+ files) - sample top 100 by importance
2. Memory spike - abort analysis, post partial results
3. LLM timeout - skip suggestion generation, post raw issues
4. GitHub API down - retry with exponential backoff, max 3 attempts

**Test Scenarios:**
- Benchmark: Analyze 50-file PR, measure end-to-end time
- Load test: Submit 20 PRs simultaneously, verify queueing
- Resource test: Monitor memory during 500-file PR
- Cost test: Measure LLM token usage, verify under budget

---

### FR-9: Multi-Language Support
**Priority:** P2 (Medium)

The system MUST support multiple programming languages:
- **Tier 1 (Full Support):** Python, TypeScript, JavaScript
- **Tier 2 (Partial Support):** Rust, Go, Java
- **Tier 3 (Syntax Only):** C, C++, Ruby, PHP

**Language-Specific Rules:**
- Python: Use ruff, mypy, bandit
- TypeScript: Use eslint, typescript compiler
- Rust: Use clippy, rustfmt
- Language-agnostic: tree-sitter for all

**Configuration:**
```toml
[review.languages.python]
enabled = true
linters = ["ruff", "mypy", "bandit"]
max_line_length = 88
complexity_threshold = 15

[review.languages.typescript]
enabled = true
linters = ["eslint"]
tsconfig_path = "tsconfig.json"
```

**Edge Cases:**
1. Unknown language - fallback to generic analysis
2. Language detection ambiguity (.h file: C or C++?) - use heuristics
3. Mixed files (Jupyter .ipynb) - extract code cells

**Test Scenarios:**
- Fixture repo with all Tier 1 languages
- Verify correct linter runs for each language
- Test cross-language smell detection (works for all)

---

### FR-10: Security and Privacy
**Priority:** P0 (Blocking)

The system MUST ensure security:
- **Authentication:** GitHub App with minimal permissions (read code, write comments)
- **Authorization:** Only analyze PRs for repos with app installed
- **Data Retention:** Delete session graphs after 30 days
- **Secret Detection:** Never log or store code containing secrets
- **Sandboxing:** Run analysis in isolated containers (no network access)

**Security Checklist:**
- [ ] GitHub webhook signature validation
- [ ] Rate limit all external API calls
- [ ] Sanitize all user input (PR titles, comments)
- [ ] Encrypt GitHub tokens at rest
- [ ] Audit log all review actions
- [ ] Implement RBAC for multi-tenant deployments

**Edge Cases:**
1. Malicious PR (code bomb, RCE attempt) - detect and reject
2. Token leak (GitHub token in code) - redact before processing
3. DoS attack (spam PRs) - rate limit per repo
4. Data breach (attacker accesses graph) - encrypt sensitive nodes

**Test Scenarios:**
- Security test suite: OWASP Top 10 test cases
- Penetration test: Attempt to inject code via PR
- Token handling: Verify encryption, rotation
- Audit: Verify all actions logged

---

## 2. NON-FUNCTIONAL REQUIREMENTS

### NFR-1: Reliability
- 99.9% uptime (SLA for webhook endpoint)
- Zero data loss (all reviews persisted to graph)
- Graceful degradation (partial analysis on errors)

### NFR-2: Maintainability
- 100% test coverage (enforced by quality gate)
- Full documentation (README, API docs, architecture diagrams)
- Monitoring dashboard (Grafana + Prometheus)

### NFR-3: Extensibility
- Plugin system for custom rules
- Webhook for external integrations
- API for querying review history

---

## 3. INTEGRATION POINTS

### GitHub API
- **Endpoint:** https://api.github.com
- **Authentication:** GitHub App JWT + Installation Token
- **Rate Limit:** 5000 requests/hour
- **Webhooks:** pull_request, pull_request_review_comment

### Tree-sitter Parsers
- **Languages:** Python, TypeScript, Rust, Go
- **Usage:** Parse AST for structural analysis

### LLM APIs
- **Providers:** Anthropic Claude, OpenAI GPT-4
- **Model Selection:** Adaptive based on complexity

### ParagonDB
- **Graph Storage:** All sessions, analyses, issues, suggestions
- **Queries:** Descendant tracking, teleology validation

---

## 4. TEST SCENARIOS

### Unit Tests
1. GitHub API client (mock responses)
2. Static analyzers (each rule independently)
3. Code smell detectors (CPG queries)
4. LLM suggestion generator (mocked LLM)
5. Graph session manager (node/edge creation)
6. Feedback classifier (sentiment analysis)

### Integration Tests
1. End-to-end: Webhook → Analysis → GitHub Comment
2. Multi-file PR: Concurrent analysis of 10 files
3. Error recovery: API failure → retry → success
4. Learning loop: Feedback → Pattern → Adjustment

### E2E Tests
1. Full workflow: Open PR → Bot reviews → Human replies → Bot learns
2. Teleology: All nodes trace to SESSION
3. Performance: 50-file PR in <60s
4. Quality gate: All tests pass, no critical issues

### Property-Based Tests
1. Any valid Python code should parse without error
2. Any analysis result should serialize/deserialize correctly
3. Graph invariants hold after any operation (DAG, teleology)

---

## 5. SUCCESS CRITERIA

### Quality Metrics (Hard Constraints)
- **Test Pass Rate:** 100%
- **Static Analysis:** 0 critical issues
- **Graph Invariants:** 100% compliance (DAG, teleology)
- **Code Coverage:** 100%
- **Security:** Pass OWASP Top 10 checklist

### Performance Metrics
- **Single File Analysis:** <500ms (p95)
- **End-to-End Review:** <60s for 50-file PR (p95)
- **Memory Usage:** <1GB per session
- **Cost per Review:** <$0.10

### User Experience Metrics
- **Suggestion Acceptance Rate:** >30%
- **False Positive Rate:** <20%
- **Time to First Comment:** <30s

---

## 6. ORCHESTRATOR GUIDANCE

This problem should exercise the full TDD pipeline:

1. **DIALECTIC Phase:**
   - Detect ambiguities: "What threshold for 'long method'?"
   - Generate questions: "Should bot post comments for info-level issues?"

2. **RESEARCH Phase:**
   - Research GitHub API best practices
   - Research code smell detection algorithms
   - Research LLM prompt engineering for code review

3. **PLAN Phase:**
   - Decompose into components: GitHubClient, StaticAnalyzer, SmellDetector, etc.
   - Build dependency graph: GitHubClient ← SessionManager ← Orchestrator

4. **BUILD Phase:**
   - Generate code for each component
   - Apply quality hooks (syntax, alignment)

5. **TEST Phase:**
   - Generate unit tests (pytest fixtures)
   - Generate integration tests (mocked GitHub API)
   - Run quality gate (100% coverage, 0 critical issues)

6. **VERIFICATION Phase:**
   - Validate teleology (all nodes trace to REQ)
   - Validate graph invariants (DAG, handshaking)
   - Generate documentation (README, architecture diagram)

**Expected Outputs:**
- 15-20 CODE nodes (one per component)
- 30-40 TEST_SUITE nodes (comprehensive coverage)
- 1 DOC node (generated README)
- Full graph with 100% teleological integrity
