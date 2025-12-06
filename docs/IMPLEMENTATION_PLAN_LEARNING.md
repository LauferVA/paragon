# Implementation Plan: Learning System Architecture

**Version:** 1.0
**Status:** PLANNED
**Parent:** [CLAUDE.md](../CLAUDE.md#6-learning-system-architecture)

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Two-Mode Solution](#2-the-two-mode-solution)
3. [AgentSignature Schema](#3-agentsignature-schema)
4. [Database Schema](#4-database-schema)
5. [Phase-Specific Confidence](#5-phase-specific-confidence)
6. [Divergence Detection](#6-divergence-detection)
7. [Forensic Analysis Engine](#7-forensic-analysis-engine)
8. [GUI Development Mode Integration](#8-gui-development-mode-integration)
9. [Learning Query Interface](#9-learning-query-interface)
10. [Integration Points](#10-integration-points)
11. [Implementation Phases](#11-implementation-phases)

---

## 1. Overview

The Learning System enables Paragon to improve its decision-making over time by:
- Tracking which agents/models succeed under which constraints
- Attributing failures to specific phases with appropriate confidence
- Detecting dangerous Test-Production divergence (false confidence)
- Learning optimal model routing based on accumulated evidence

**Key Insight:** Study Mode gathers clean, unbiased data. Production Mode uses learned priors. Never mix them.

---

## 2. The Two-Mode Solution

### 2.1 Mode Definitions

| Mode | Purpose | Behavior |
|------|---------|----------|
| **STUDY** | Cold-start data collection | Random model selection, full logging, no biasing |
| **PRODUCTION** | Optimized operation | Learned priors, model routing, cost optimization |

### 2.2 Transition Criteria

Study → Production transition requires:
- Minimum 100 sessions completed
- Statistical significance (p < 0.05) on at least 3 constraint dimensions
- Human review of learned priors

### 2.3 Implementation

**File:** `infrastructure/learning.py`

```python
from enum import Enum

class LearningMode(str, Enum):
    STUDY = "study"
    PRODUCTION = "production"

class LearningSystem:
    def __init__(self, mode: LearningMode = LearningMode.STUDY):
        self.mode = mode
        self.exploration_rate = 0.1  # Even in production, explore 10%

    def can_transition_to_production(self) -> TransitionReport:
        """Check if sufficient data exists for production mode."""
        session_count = TrainingStore.get_session_count()
        significant_dimensions = TrainingStore.get_significant_dimensions(p_threshold=0.05)

        return TransitionReport(
            ready=session_count >= 100 and len(significant_dimensions) >= 3,
            session_count=session_count,
            significant_dimensions=significant_dimensions,
            requires_human_review=True,
        )
```

---

## 3. SignatureChain Schema

Every node carries a complete lifecycle history - who worked on it, when, in what phase, and what action they took.

**File:** `core/schemas.py`

```python
import msgspec
from enum import Enum
from typing import Dict, Any
import uuid

class CyclePhase(str, Enum):
    DIALECTIC = "dialectic"
    PLAN = "plan"
    BUILD = "build"
    TEST = "test"

class SignatureAction(str, Enum):
    """What the agent did to the node."""
    CREATED = "created"         # Initial creation
    MODIFIED = "modified"       # Content changed
    VERIFIED = "verified"       # Passed verification
    REJECTED = "rejected"       # Failed verification
    ESCALATED = "escalated"     # Escalated to user/higher agent
    SUPERSEDED = "superseded"   # Replaced by another approach

class NodeOutcome(str, Enum):
    """The fate of a node, determined post-run."""
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"
    TEST_PROD_DIVERGENCE = "test_prod_divergence"
    UNEXERCISED = "unexercised"
    INDETERMINATE = "indeterminate"

class AgentSignature(msgspec.Struct, frozen=True):
    """Single contribution record in a node's history."""
    agent_id: str              # e.g., "architect_agent"
    model_id: str              # e.g., "claude-3-5-sonnet"
    phase: CyclePhase          # Which phase this occurred in
    action: SignatureAction    # What the agent did
    temperature: float         # 0.0 - 1.0
    context_constraints: Dict[str, Any]  # User prefs active at this moment
    timestamp: str             # ISO 8601
    notes: str = ""            # Optional: why this action was taken

class SignatureChain(msgspec.Struct):
    """
    Full lifecycle history of a node (Law 9).

    Tracks all agent contributions across all phases, including
    incorrect approaches and fixes for learning purposes.
    """
    node_id: str                           # Permanent ID
    state_id: str                          # Changes on evolution (UUID)
    signatures: list[AgentSignature]       # Ordered history
    is_replacement: bool = False           # True if replaced another node
    replaced_node_id: str | None = None    # ID of replaced node (for lineage)

    def add_signature(self, sig: AgentSignature) -> "SignatureChain":
        """Add a new signature and generate new state_id."""
        return SignatureChain(
            node_id=self.node_id,
            state_id=str(uuid.uuid4()),  # New state on any change
            signatures=[*self.signatures, sig],
            is_replacement=self.is_replacement,
            replaced_node_id=self.replaced_node_id,
        )

    def get_phase_signatures(self, phase: CyclePhase) -> list[AgentSignature]:
        """Get all signatures from a specific phase."""
        return [s for s in self.signatures if s.phase == phase]

    def get_agent_signatures(self, agent_id: str) -> list[AgentSignature]:
        """Get all signatures from a specific agent."""
        return [s for s in self.signatures if s.agent_id == agent_id]

    @property
    def creation_signature(self) -> AgentSignature | None:
        """Get the original creation signature."""
        for sig in self.signatures:
            if sig.action == SignatureAction.CREATED:
                return sig
        return None

    @property
    def latest_signature(self) -> AgentSignature | None:
        """Get the most recent signature."""
        return self.signatures[-1] if self.signatures else None
```

### 3.1 Evolution vs Replacement

**Critical for maintaining graph acyclicity:**

| Scenario | Behavior | Graph Impact |
|----------|----------|--------------|
| **Evolution** | Same `node_id`, new `state_id`, append signature | No structural change, edges preserved |
| **Replacement** | New `node_id`, `is_replacement=True`, link via `replaced_node_id` | Old node marked SUPERSEDED, new edges created |

```python
def evolve_node(node: GraphNode, signature: AgentSignature) -> GraphNode:
    """
    Evolve an existing node (same ID, new state).
    Used when content changes but the node's role in the graph is unchanged.
    """
    new_chain = node.signature_chain.add_signature(signature)
    return GraphNode(
        id=node.id,  # Same ID
        content=new_content,
        type=node.type,
        signature_chain=new_chain,
    )

def replace_node(old_node: GraphNode, new_content: Any, signature: AgentSignature) -> GraphNode:
    """
    Replace a node entirely (new ID, marks old as superseded).
    Used when the approach was wrong and needs complete replacement.
    """
    # Mark old node as superseded
    supersede_sig = AgentSignature(
        agent_id=signature.agent_id,
        model_id=signature.model_id,
        phase=signature.phase,
        action=SignatureAction.SUPERSEDED,
        temperature=signature.temperature,
        context_constraints=signature.context_constraints,
        timestamp=get_timestamp(),
        notes=f"Replaced by new node",
    )
    old_node.signature_chain.add_signature(supersede_sig)

    # Create new replacement node
    new_chain = SignatureChain(
        node_id=generate_node_id(),  # New ID
        state_id=str(uuid.uuid4()),
        signatures=[signature],
        is_replacement=True,
        replaced_node_id=old_node.id,
    )

    return GraphNode(
        id=new_chain.node_id,
        content=new_content,
        type=old_node.type,
        signature_chain=new_chain,
    )
```

### 3.2 Tracking Incorrect Approaches

For learning and debugging, we track wrong turns:

```python
# Example: Builder tries approach A, fails, tries approach B

# Attempt 1: Create with approach A
sig1 = AgentSignature(
    agent_id="builder_agent",
    action=SignatureAction.CREATED,
    phase=CyclePhase.BUILD,
    notes="Initial implementation using recursion",
    ...
)

# Attempt 1: Tester rejects
sig2 = AgentSignature(
    agent_id="tester_agent",
    action=SignatureAction.REJECTED,
    phase=CyclePhase.TEST,
    notes="Stack overflow on large inputs",
    ...
)

# Attempt 2: Builder modifies to approach B
sig3 = AgentSignature(
    agent_id="builder_agent",
    action=SignatureAction.MODIFIED,
    phase=CyclePhase.BUILD,
    notes="Switched to iterative approach",
    ...
)

# Attempt 2: Tester verifies
sig4 = AgentSignature(
    agent_id="tester_agent",
    action=SignatureAction.VERIFIED,
    phase=CyclePhase.TEST,
    notes="All tests pass",
    ...
)

# The SignatureChain now contains [sig1, sig2, sig3, sig4]
# This history is invaluable for learning why recursion failed here
```

### 3.3 Enforcement

**File:** `agents/tools.py` - modify `add_node_safe()`

```python
def add_node_safe(node_type: str, content: Any,
                  signature_chain: SignatureChain) -> NodeResult:
    """
    Add a node with mandatory signature chain.

    Law 9: Every node MUST have a SignatureChain with at least one signature.
    """
    if signature_chain is None:
        return NodeResult(
            success=False,
            node_id="",
            message="Law 9 violation: signature_chain is required"
        )

    if not signature_chain.signatures:
        return NodeResult(
            success=False,
            node_id="",
            message="Law 9 violation: signature_chain must have at least one signature"
        )

    # Validate creation signature exists for new nodes
    if not signature_chain.is_replacement:
        if signature_chain.creation_signature is None:
            return NodeResult(
                success=False,
                node_id="",
                message="Law 9 violation: new nodes must have CREATED signature"
            )

    # ... existing validation logic ...
```

### 3.4 Query Patterns

```python
# Find all nodes a specific agent worked on
def get_nodes_by_agent(graph: ExecutionGraph, agent_id: str) -> list[GraphNode]:
    return [
        node for node in graph.get_all_nodes()
        if any(s.agent_id == agent_id for s in node.signature_chain.signatures)
    ]

# Find nodes that were rejected and then fixed
def get_fix_patterns(graph: ExecutionGraph) -> list[tuple[GraphNode, list[AgentSignature]]]:
    patterns = []
    for node in graph.get_all_nodes():
        chain = node.signature_chain.signatures
        for i, sig in enumerate(chain):
            if sig.action == SignatureAction.REJECTED:
                # Found rejection - get the fix that followed
                fixes = [s for s in chain[i+1:] if s.action in (
                    SignatureAction.MODIFIED, SignatureAction.VERIFIED
                )]
                if fixes:
                    patterns.append((node, [sig, *fixes]))
    return patterns

# Get replacement lineage (for understanding why approaches were abandoned)
def get_replacement_chain(graph: ExecutionGraph, node_id: str) -> list[GraphNode]:
    """Walk backwards through replacements to see evolution of approach."""
    chain = []
    current_id = node_id
    while current_id:
        node = graph.get_node(current_id)
        if node:
            chain.append(node)
            current_id = node.signature_chain.replaced_node_id
        else:
            break
    return list(reversed(chain))  # Oldest first
```

---

## 4. Database Schema

### 4.1 Technology Selection

**Current Stack** (msgspec + SQLite):

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Persistence** | SQLite | Battle-tested, zero-config, ACID, sufficient until proven otherwise |
| **Serialization** | msgspec | 10-80x faster than alternatives, pure Python, schema-flexible |
| **Graph Compute** | rustworkx | Already Rust-accelerated where it matters |
| **Data Analysis** | polars | Already Rust-accelerated where it matters |

**Architectural Principle:** Use Rust-accelerated *libraries* (rustworkx, polars), not custom Rust *code* (PyO3). This provides 80% of performance gains with 20% of complexity.

**Why NOT PyO3/zero-copy (Decision: 2025-12-05):**
1. **Premature Optimization** - No proven bottleneck in current stack
2. **Schema Rigidity** - Any schema change requires Rust recompile (schema is still evolving)
3. **Dependency Hell** - pyo3 + polars-core + pyarrow version alignment is fragile
4. **Contributor Friction** - Requires Rust toolchain for all contributors

### 4.2 Normalized SQLite Tables

**File:** `infrastructure/training_store.py`

```sql
-- Core tables (normalized for clean queries)

CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    global_outcome TEXT CHECK(global_outcome IN ('success', 'failure', 'divergence')),
    failure_code TEXT CHECK(failure_code IN ('F1', 'F2', 'F3', 'F4', 'F5', NULL)),
    user_feedback TEXT,
    learning_mode TEXT CHECK(learning_mode IN ('study', 'production'))
);

-- Signature chain tracking (Law 9)
CREATE TABLE signature_chains (
    chain_id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    state_id TEXT NOT NULL,           -- Changes on evolution
    is_replacement BOOLEAN DEFAULT FALSE,
    replaced_node_id TEXT,            -- Links to superseded node
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(node_id, state_id)
);

CREATE TABLE signatures (
    signature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id INTEGER REFERENCES signature_chains(chain_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    phase TEXT NOT NULL CHECK(phase IN ('dialectic', 'plan', 'build', 'test')),
    action TEXT NOT NULL CHECK(action IN (
        'created', 'modified', 'verified', 'rejected', 'escalated', 'superseded'
    )),
    temperature REAL NOT NULL,
    notes TEXT,
    timestamp DATETIME NOT NULL,
    sequence INTEGER NOT NULL,        -- Order within chain
    UNIQUE(chain_id, sequence)
);

CREATE TABLE signature_constraints (
    signature_id INTEGER REFERENCES signatures(signature_id) ON DELETE CASCADE,
    constraint_key TEXT NOT NULL,
    constraint_value TEXT NOT NULL,
    PRIMARY KEY (signature_id, constraint_key)
);

CREATE TABLE node_attributions (
    attribution_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    chain_id INTEGER REFERENCES signature_chains(chain_id),
    outcome TEXT NOT NULL CHECK(outcome IN (
        'verified_success', 'verified_failure',
        'test_prod_divergence', 'unexercised', 'indeterminate'
    )),
    confidence REAL DEFAULT 1.0 CHECK(confidence >= 0.0 AND confidence <= 1.0),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME,
    version INTEGER DEFAULT 1
);

CREATE TABLE divergence_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    test_outcome TEXT,
    prod_outcome TEXT,
    divergence_type TEXT CHECK(divergence_type IN ('false_positive', 'false_negative')),
    severity TEXT CHECK(severity IN ('critical', 'major', 'minor')),
    root_cause TEXT,
    logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    analyzed_at DATETIME
);

-- Replacement lineage tracking
CREATE TABLE replacement_lineage (
    lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    current_node_id TEXT NOT NULL,
    replaced_node_id TEXT NOT NULL,
    replacement_reason TEXT,
    replaced_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast queries
CREATE INDEX idx_runs_outcome ON runs(global_outcome);
CREATE INDEX idx_runs_mode ON runs(learning_mode);
CREATE INDEX idx_chains_node ON signature_chains(node_id);
CREATE INDEX idx_chains_replaced ON signature_chains(replaced_node_id);
CREATE INDEX idx_sigs_chain ON signatures(chain_id);
CREATE INDEX idx_sigs_agent ON signatures(agent_id);
CREATE INDEX idx_sigs_action ON signatures(action);
CREATE INDEX idx_sigs_phase ON signatures(phase);
CREATE INDEX idx_attr_outcome ON node_attributions(outcome);
CREATE INDEX idx_attr_run ON node_attributions(run_id);
CREATE INDEX idx_divergence_type ON divergence_events(divergence_type);
CREATE INDEX idx_lineage_current ON replacement_lineage(current_node_id);
CREATE INDEX idx_lineage_replaced ON replacement_lineage(replaced_node_id);
```

### 4.3 Query Examples

```sql
-- Get full signature history for a node
SELECT s.*, sc.constraint_key, sc.constraint_value
FROM signature_chains c
JOIN signatures s ON s.chain_id = c.chain_id
LEFT JOIN signature_constraints sc ON sc.signature_id = s.signature_id
WHERE c.node_id = ?
ORDER BY s.sequence;

-- Find nodes that were rejected then fixed (learning patterns)
SELECT DISTINCT c.node_id,
    s1.notes as rejection_reason,
    s2.notes as fix_description
FROM signatures s1
JOIN signatures s2 ON s1.chain_id = s2.chain_id AND s2.sequence > s1.sequence
JOIN signature_chains c ON c.chain_id = s1.chain_id
WHERE s1.action = 'rejected'
  AND s2.action IN ('modified', 'verified');

-- Get replacement lineage for a node
WITH RECURSIVE lineage AS (
    SELECT node_id, replaced_node_id, 0 as depth
    FROM signature_chains WHERE node_id = ?
    UNION ALL
    SELECT c.node_id, c.replaced_node_id, l.depth + 1
    FROM signature_chains c
    JOIN lineage l ON c.node_id = l.replaced_node_id
)
SELECT * FROM lineage ORDER BY depth DESC;

-- Agent contribution analysis
SELECT agent_id, phase, action, COUNT(*) as count,
    AVG(CASE WHEN na.outcome = 'verified_success' THEN 1.0 ELSE 0.0 END) as success_rate
FROM signatures s
JOIN signature_chains c ON c.chain_id = s.chain_id
LEFT JOIN node_attributions na ON na.node_id = c.node_id
GROUP BY agent_id, phase, action;
```

### 4.4 TrainingStore Class

```python
class TrainingStore:
    DB_PATH = Path("data/training.db")

    @classmethod
    def initialize(cls):
        """Create tables if they don't exist."""
        cls.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(cls.DB_PATH) as conn:
            conn.executescript(SCHEMA_SQL)

    @classmethod
    def log_attributions(cls, run_id: str, attributions: list, upsert: bool = False):
        """
        Log node attributions. If upsert=True, update existing records.
        """
        with sqlite3.connect(cls.DB_PATH) as conn:
            for attr in attributions:
                sig_id = cls._get_or_create_signature(conn, attr.signature)

                if upsert:
                    conn.execute("""
                        INSERT INTO node_attributions
                        (run_id, node_id, signature_id, outcome, confidence, updated_at, version)
                        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, 1)
                        ON CONFLICT(run_id, node_id) DO UPDATE SET
                            outcome = excluded.outcome,
                            confidence = excluded.confidence,
                            updated_at = CURRENT_TIMESTAMP,
                            version = version + 1
                    """, (run_id, attr.node_id, sig_id, attr.outcome.value, attr.confidence))
                else:
                    conn.execute("""
                        INSERT INTO node_attributions
                        (run_id, node_id, signature_id, outcome, confidence)
                        VALUES (?, ?, ?, ?, ?)
                    """, (run_id, attr.node_id, sig_id, attr.outcome.value, attr.confidence))
```

---

## 5. Phase-Specific Confidence

Different phases have different attribution confidence based on causal proximity to failures.

| Phase | Confidence | Rationale |
|-------|------------|-----------|
| **DIALECTIC** | 0.6 | Topology flaws may not manifest until BUILD |
| **PLAN** | 0.7 | Architecture issues may be implementation-dependent |
| **BUILD** | 0.9 | Direct causal link to code failures |
| **TEST** | 0.8 | Test quality affects verification accuracy |

```python
PHASE_CONFIDENCE = {
    CyclePhase.DIALECTIC: 0.6,
    CyclePhase.PLAN: 0.7,
    CyclePhase.BUILD: 0.9,
    CyclePhase.TEST: 0.8,
}
```

---

## 6. Divergence Detection

> **The most dangerous state: tests pass but production fails.**

**File:** `infrastructure/divergence.py`

```python
class DivergenceDetector:
    """Detects Test-Production divergence - the 'false confidence' state."""

    def check_divergence(self, run_id: str, test_result: bool,
                         user_feedback: str) -> DivergenceEvent | None:
        """Called when user provides feedback after a 'passing' run."""

        if test_result and user_feedback in ("rejected", "broken", "wrong"):
            # CRITICAL: Tests lied to us
            return DivergenceEvent(
                run_id=run_id,
                divergence_type="false_positive",
                severity=self._assess_severity(user_feedback),
            )

        if not test_result and user_feedback in ("actually_works", "acceptable"):
            # Tests were too strict
            return DivergenceEvent(
                run_id=run_id,
                divergence_type="false_negative",
                severity="minor",
            )

        return None

    def _assess_severity(self, feedback: str) -> str:
        """Assess severity based on user feedback language."""
        if feedback in ("broken", "critical", "unusable"):
            return "critical"
        if feedback in ("wrong", "incorrect"):
            return "major"
        return "minor"
```

### 6.1 Divergence Metrics

```python
# Target metrics
divergence_rate = false_positives / total_passes  # Target: < 5%
test_strictness = false_negatives / total_failures  # Target: < 10%
```

---

## 7. Forensic Analysis Engine

**File:** `infrastructure/attribution.py`

```python
class ForensicAnalyzer:
    """
    Post-run attribution engine. Re-runnable for iterative improvement.
    """

    def __init__(self):
        self.divergence_detector = DivergenceDetector()

    def analyze(self, run_id: str, graph: ExecutionGraph, report: RunReport,
                rerun: bool = False) -> AttributionReport:
        """
        Perform forensic analysis on a completed run.

        Args:
            run_id: Unique run identifier
            graph: The execution graph with all nodes
            report: Test results, coverage, user feedback
            rerun: If True, update existing attributions (increment version)

        Returns:
            AttributionReport with all node outcomes
        """
        # 1. Determine global outcome
        global_outcome = self._determine_global_outcome(report)

        # 2. Check for divergence (if user feedback available)
        divergence = None
        if report.user_feedback and report.tests_passed:
            divergence = self.divergence_detector.check_divergence(
                run_id, report.tests_passed, report.user_feedback
            )
            if divergence:
                global_outcome = "divergence"
                TrainingStore.log_divergence(divergence)

        # 3. Attribute ALL phases (not just BUILD)
        attributions = []
        for phase in CyclePhase:
            phase_nodes = graph.get_nodes_by_phase(phase)
            confidence = PHASE_CONFIDENCE[phase]

            for node in phase_nodes:
                if not node.author_signature:
                    raise ValueError(f"Node {node.id} missing signature - Law 9 violation")

                outcome = self._determine_node_outcome(
                    node, global_outcome, report, phase
                )

                attributions.append(NodeAttribution(
                    node_id=node.id,
                    signature=node.author_signature,
                    outcome=outcome,
                    confidence=confidence,
                ))

        # 4. Commit to database (upsert if rerun)
        TrainingStore.log_attributions(run_id, attributions, upsert=rerun)

        return AttributionReport(
            run_id=run_id,
            global_outcome=global_outcome,
            attributions=attributions,
            divergence_detected=divergence is not None,
            version=self._get_analysis_version(run_id),
        )

    def _determine_node_outcome(self, node, global_outcome, report, phase) -> NodeOutcome:
        """Determine individual node outcome with phase-specific logic."""

        if global_outcome == "divergence":
            return NodeOutcome.TEST_PROD_DIVERGENCE

        if global_outcome == "success":
            return NodeOutcome.VERIFIED_SUCCESS

        # Global failure - attribute based on evidence
        if node.id in report.failure_trace:
            return NodeOutcome.VERIFIED_FAILURE

        # Phase-specific heuristics
        if phase == CyclePhase.DIALECTIC and report.failure_code == "F1":
            return NodeOutcome.VERIFIED_FAILURE

        if phase == CyclePhase.TEST and report.failure_code == "F3":
            return NodeOutcome.VERIFIED_FAILURE

        return NodeOutcome.INDETERMINATE
```

---

## 8. GUI Development Mode Integration

> **Forensic re-analysis is most powerful when combined with side-by-side comparison in GUI development mode.**

### 8.1 Use Case: Iterative Improvement

When a developer is investigating why a run failed or produced unexpected results:

1. **Load run in Development Mode** - Side-by-side view of graph state vs expected
2. **Trigger forensic re-analysis** - With new user feedback or updated heuristics
3. **Compare attribution versions** - See how blame shifted between analyses
4. **Identify patterns** - Spot systematic issues across multiple runs

### 8.2 Side-by-Side Comparison Panel

```
┌─────────────────────────────────────────────────────────────────────┐
│  FORENSIC ANALYSIS COMPARISON                    [Version 1 → 2]   │
├─────────────────────────┬───────────────────────────────────────────┤
│  ORIGINAL ANALYSIS      │  RE-ANALYSIS                              │
│  (2025-12-06 14:32)     │  (2025-12-06 16:45)                       │
├─────────────────────────┼───────────────────────────────────────────┤
│  Node: plan_auth_flow   │  Node: plan_auth_flow                     │
│  Outcome: INDETERMINATE │  Outcome: VERIFIED_FAILURE ← CHANGED      │
│  Confidence: 0.7        │  Confidence: 0.7                          │
│                         │  Reason: User feedback "auth broken"      │
├─────────────────────────┼───────────────────────────────────────────┤
│  Node: build_jwt_token  │  Node: build_jwt_token                    │
│  Outcome: VERIFIED_FAIL │  Outcome: VERIFIED_FAILURE                │
│  Confidence: 0.9        │  Confidence: 0.9                          │
├─────────────────────────┼───────────────────────────────────────────┤
│  DIVERGENCE: None       │  DIVERGENCE: false_positive (critical)    │
└─────────────────────────┴───────────────────────────────────────────┘
```

### 8.3 Integration with RT Visualization

**File:** `api/websocket_forensic.py`

```python
@router.websocket("/ws/forensic/{run_id}")
async def forensic_ws(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for forensic analysis in dev mode."""
    await websocket.accept()

    while True:
        message = await websocket.receive_json()

        if message["action"] == "reanalyze":
            # Trigger re-analysis with new feedback
            report = await forensic_analyzer.analyze(
                run_id=run_id,
                graph=await load_graph(run_id),
                report=RunReport(
                    user_feedback=message.get("user_feedback"),
                    **message.get("report_updates", {})
                ),
                rerun=True
            )

            # Send comparison data
            await websocket.send_json({
                "type": "comparison",
                "previous_version": report.version - 1,
                "current_version": report.version,
                "changes": compute_attribution_diff(run_id, report.version),
            })

        elif message["action"] == "get_history":
            # Return all analysis versions for this run
            history = TrainingStore.get_attribution_history(run_id)
            await websocket.send_json({
                "type": "history",
                "versions": history,
            })
```

### 8.4 Development Mode Features

| Feature | Description |
|---------|-------------|
| **Version Slider** | Scrub through forensic analysis versions |
| **Diff Highlighting** | Changed attributions highlighted in orange |
| **Blame Flow** | Animated visualization of blame propagation |
| **Pattern Detection** | Auto-detect nodes that frequently flip outcome |
| **Export to Training** | Mark corrected attributions as ground truth |

**Reference:** [RESEARCH_RT_VISUALIZATION.md](RESEARCH_RT_VISUALIZATION.md#5-gui-component-designs)

---

## 9. Learning Query Interface

```python
class LearningSystem:
    """Query interface for Production Mode decisions."""

    def get_success_rate(self, model_id: str, phase: str,
                         constraints: dict = None) -> float:
        """
        Returns P(success | model, phase, constraints).
        Only available in PRODUCTION mode with sufficient data.
        """
        query = """
            SELECT
                COUNT(CASE WHEN na.outcome = 'verified_success' THEN 1 END) as successes,
                COUNT(*) as total
            FROM node_attributions na
            JOIN signatures s ON na.signature_id = s.signature_id
            WHERE s.model_id = ? AND s.phase = ?
        """
        params = [model_id, phase]

        if constraints:
            # Add constraint filtering via node_constraints join
            query += """
                AND na.attribution_id IN (
                    SELECT nc.attribution_id
                    FROM node_constraints nc
                    JOIN constraints c ON nc.constraint_id = c.constraint_id
                    WHERE c.constraint_key = ? AND c.constraint_value = ?
                )
            """
            for key, value in constraints.items():
                params.extend([key, str(value)])

        with sqlite3.connect(TrainingStore.DB_PATH) as conn:
            result = conn.execute(query, params).fetchone()
            successes, total = result

        return successes / total if total > 0 else 0.5  # Prior

    def get_recommendation(self, task_context: TaskContext) -> ModelRecommendation | None:
        """
        Returns model recommendation based on learned priors.
        Only biases decisions in PRODUCTION mode.
        """
        if self.mode == LearningMode.STUDY:
            return None  # No biasing in study mode

        candidates = []
        for model in self.available_models:
            success_rate = self.get_success_rate(
                model.id,
                task_context.phase.value,
                task_context.constraints
            )
            candidates.append((model, success_rate))

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            return random.choice(candidates)[0]

        return max(candidates, key=lambda x: x[1])[0]
```

---

## 10. Integration Points

### 10.1 Files to Modify

| File | Changes |
|------|---------|
| `core/schemas.py` | Add `AgentSignature`, `NodeOutcome`, `CyclePhase` |
| `core/graph_db.py` | Add `get_nodes_by_phase()` method |
| `agents/tools.py` | Modify `add_node_safe()` to require signature |
| `agents/orchestrator.py` | Sign nodes at creation, call forensic analysis |
| `benchmarks/harness.py` | Return `RunReport` with failure_trace |

### 10.2 New Files

| File | Purpose |
|------|---------|
| `infrastructure/training_store.py` | SQLite backend with normalized schema |
| `infrastructure/attribution.py` | Forensic analysis engine |
| `infrastructure/divergence.py` | Test-Production divergence detection |
| `infrastructure/learning.py` | Learning mode management and queries |
| `api/websocket_forensic.py` | WebSocket for dev mode forensics |

---

## 11. Implementation Phases

| Phase | Goal | Deliverable | Depends On |
|-------|------|-------------|------------|
| **L0** | Schema additions | `AgentSignature` in schemas, enforcement in hooks | - |
| **L1** | Database setup | `training_store.py` with normalized schema | L0 |
| **L2** | Forensic engine | `attribution.py` with all-phase analysis | L1 |
| **L3** | Divergence detection | `divergence.py` with logging | L2 |
| **L4** | Study mode | Full logging, no biasing, data collection | L3 |
| **L5** | Production mode | Learned priors, model routing | L4 |
| **L6** | GUI dev mode | Side-by-side forensic comparison | L2, RT-Viz |

---

## 12. Phase L∞: Performance Escape Hatches

> **Status:** DOCUMENTED, NOT IMPLEMENTED
> **Decision Date:** 2025-12-05
> **Trigger:** Only after benchmarks prove SQLite bottleneck at >500K nodes

### 12.1 Why NOT Now

The following options were researched and explicitly **rejected** for the current phase:

| Technology | Rejected Reason |
|------------|-----------------|
| **PyO3 + Zero-Copy** | Schema rigidity (Rust recompile on any schema change), dependency hell (pyo3 + polars-core + pyarrow version alignment) |
| **redb** | Premature optimization - no proven bottleneck |
| **fjall** | Premature optimization - no proven bottleneck |

**Architectural Principle:** Use Rust-accelerated *libraries* (rustworkx, polars), not custom Rust *code* (PyO3). This provides 80% of performance gains with 20% of complexity.

### 12.2 Escape Hatch Options (If Needed)

If benchmarks prove SQLite is a bottleneck, these options remain available:

#### Option A: redb (Pure Rust ACID)
```toml
# Future Cargo.toml (if needed)
[dependencies]
redb = "2.4"
pyo3 = { version = "0.23", features = ["extension-module"] }
```

**Characteristics:**
- Pure Rust, zero unsafe code
- Single-writer MVCC (7x faster than SQLite for individual writes)
- 2.7x faster than sled for point lookups
- Best for: Graph state storage with high write frequency

#### Option B: fjall (Rust LSM)
```toml
# Future Cargo.toml (if needed)
[dependencies]
fjall = "2.5"
pyo3 = { version = "0.23", features = ["extension-module"] }
```

**Characteristics:**
- LSM-tree with keyspace partitioning
- Best batch write performance
- Built-in compression
- Best for: Append-heavy provenance logs

#### Option C: Zero-Copy Arrow IPC
```python
# Future bridge pattern (if needed)
import pyarrow as pa
from paragon_store import get_batch  # PyO3 binding

# Rust side returns Arrow IPC bytes
ipc_bytes = get_batch(query)
reader = pa.ipc.open_stream(ipc_bytes)
batch = reader.read_next_batch()
```

**Characteristics:**
- Zero deserialization overhead
- Requires schema stability (any change = Rust recompile)
- Best for: High-volume read paths after schema freeze

### 12.3 W3C PROV-AGENT Standard (Reference)

For future provenance interoperability:

```python
# PROV-O compatible serialization (msgspec implementation)
class ProvActivity(msgspec.Struct, kw_only=True, frozen=True):
    """W3C PROV-O Activity for agent actions."""
    prov_type: str = "prov:Activity"
    prov_id: str
    prov_startedAtTime: str
    prov_endedAtTime: str
    prov_wasAssociatedWith: str  # Agent URI
    prov_used: list[str]  # Entity URIs
    prov_generated: list[str]  # Entity URIs
```

---

## 13. Benchmarking Requirements

> **Mandate:** No optimization work proceeds without benchmark evidence.

### 13.1 Required Benchmarks Before Any L∞ Work

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| **Node Count Scale** | 500K nodes | Time to `get_waves()` |
| **Write Throughput** | 10K nodes/sec | SQLite insert rate |
| **Query Latency** | p99 < 100ms | Attribution query time |
| **Memory Footprint** | <4GB RSS | Peak memory at 500K nodes |
| **Cold Start** | <5s | Time to load 500K node graph |

### 13.2 Benchmark Protocol

```python
# benchmarks/protocol_omega.py - Scale test
def benchmark_scale_test():
    """
    Run BEFORE any L∞ escape hatch implementation.

    1. Generate synthetic graph with 500K nodes
    2. Measure all targets in 13.1
    3. If ALL targets pass: SQLite is sufficient, no L∞ needed
    4. If ANY target fails: Document which target, propose L∞ solution
    """
    pass  # Implementation when needed
```

### 13.3 Decision Framework

```
IF all benchmarks pass at 500K nodes:
    → SQLite remains the stack
    → Re-benchmark at 1M nodes if/when reached

IF any benchmark fails:
    → Document specific bottleneck
    → Propose minimal L∞ intervention
    → Get explicit approval before implementing
    → Schema MUST be stable before any PyO3 work
```

---

## References

- [CLAUDE.md Section 6](../CLAUDE.md#6-learning-system-architecture) - Parent specification
- [RESEARCH_RT_VISUALIZATION.md](RESEARCH_RT_VISUALIZATION.md) - GUI technology stack
- [RESEARCH_ADAPTIVE_QUESTIONING.md](RESEARCH_ADAPTIVE_QUESTIONING.md) - Question learning context
- [redb GitHub](https://github.com/cberner/redb) - Pure Rust ACID database
- [fjall GitHub](https://github.com/fjall-rs/fjall) - Rust LSM-tree storage
- [W3C PROV-O](https://www.w3.org/TR/prov-o/) - Provenance ontology standard
