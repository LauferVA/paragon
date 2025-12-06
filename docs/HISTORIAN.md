# The Historian Module

**Theme 1 - Lifecycle, Governance, & Git**

The Historian module provides three interconnected capabilities for tracking, versioning, and documenting the evolution of the Paragon graph:

1. **GitSync** - Automatic Git commits on transaction boundaries
2. **Documenter** - Auto-generated documentation from graph state
3. **AuditLogger** - Forensic traceability for all mutations

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   PARAGON GRAPH                         │
│  (REQ → SPEC → CODE with teleology chains)             │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Mutations
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   HISTORIAN MODULE                      │
├─────────────────────────────────────────────────────────┤
│  1. AuditLogger     → data/audit.log (JSONL)           │
│  2. GitSync         → .git/ (semantic commits)          │
│  3. Documenter      → README.md, docs/wiki/            │
└─────────────────────────────────────────────────────────┘
```

**Design Principles:**
- **Graph-Native**: Documentation reflects the graph, not the prompts
- **Transaction Boundaries**: Commits happen on logical units, not every token
- **Semantic Messages**: Derived from teleology chains (CODE → SPEC → REQ)
- **Forensic Traceability**: Every action has: who, what, when, state

---

## 1. GitSync

### Purpose
Auto-triggers Git commits on graph transaction boundaries with semantic messages derived from the Teleology Chain.

### Key Features
- **Transaction Boundary Detection**: Commits when logical units complete (e.g., Node + Edge creation)
- **Semantic Commit Messages**: Follows [Conventional Commits](https://www.conventionalcommits.org/)
  - Format: `<type>: [REQ-<id>] <description>`
  - Example: `feat: [REQ-abc123] Implement hash function`
- **Agent Attribution**: Tags commits with `agent_id` that authored the change
- **Teleology Traversal**: Walks CODE → SPEC → REQ to generate meaningful messages

### Configuration

In `config/paragon.toml`:

```toml
[git]
enabled = true              # Enable/disable git sync
repo_path = "."             # Repository path
auto_commit = true          # Auto-commit on transactions
auto_push = false           # Auto-push (dangerous!)
commit_prefix = ""          # Optional prefix
author_name = "Paragon"     # Git author
author_email = "paragon@localhost"
```

### Usage

```python
from infrastructure.git_sync import GitSync

# Initialize with database reference (for teleology)
git_sync = GitSync(db=paragon_db)

# Trigger on transaction complete
success = git_sync.on_transaction_complete(
    nodes_created=["node_123", "node_456"],
    edges_created=[("node_123", "spec_789", "IMPLEMENTS")],
    agent_id="builder-gpt4-v1",
    agent_role="BUILDER"
)
```

### Hook Pattern

Integrate with `add_node_safe` callback:

```python
def on_add_node_success(node_id: str, node_type: str):
    """Called after successful node creation."""
    # Track for transaction
    transaction.nodes_created.append(node_id)

    # On transaction boundary (e.g., after edge added)
    git_sync.on_transaction_complete(
        nodes_created=transaction.nodes_created,
        edges_created=transaction.edges_created,
        agent_id=current_agent.id
    )
```

### Commit Message Generation

The semantic message is derived by:

1. **Identify Primary Node**: Use first created node in transaction
2. **Traverse Teleology**: Walk CODE → SPEC → REQ
3. **Determine Type**: Based on node type (CODE = feat, SPEC = docs, etc.)
4. **Extract REQ ID**: Use short hash of REQ node
5. **Generate Description**: From node content (first line)

Example:
```
feat: [REQ-8e6243b6] Implement hash function

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
Agent-ID: builder-gpt4-v1
Agent-Role: BUILDER
```

---

## 2. Documenter Agent

### Purpose
Background agent that generates documentation from the graph state, reflecting what IS, not what SHOULD BE.

### Key Features

1. **Auto-README**: Complete rewrite of `README.md` on every major cycle
   - Queries REQ, SPEC, CODE nodes
   - Groups by file path
   - Shows status with emojis (✅ verified, ⏳ processing, etc.)

2. **The Wiki**: Generates Markdown files in `docs/wiki/` for verified SPEC nodes
   - One page per SPEC
   - Includes parent REQ, implementing CODE
   - Full teleology chain

3. **Changelog**: Appends to `CHANGELOG.md` by diffing Merkle Root between commits
   - Tracks graph evolution
   - Records node/edge counts
   - Append-only for historical record

### Configuration

In `config/paragon.toml`:

```toml
[documenter]
readme_path = "README.md"
changelog_path = "CHANGELOG.md"
wiki_path = "docs/wiki"
auto_generate = true
include_pending_nodes = false
```

### Usage

```python
from agents.documenter import Documenter

documenter = Documenter(db=paragon_db)

# Generate all documentation
documenter.generate_readme()    # Complete overwrite
documenter.generate_wiki()      # One page per SPEC
documenter.append_changelog(    # Append-only
    old_merkle="abc123",
    new_merkle="def456",
    description="Initial implementation"
)
```

### Example README Output

```markdown
# Project Paragon
*Graph-native AI software platform - Auto-generated from graph state*

## Overview
This project contains 1 requirement(s):

✅ **REQ-8e6243b6**: Implement a cryptographic hash function module

## Specifications
| ID | Status | Description | Requirement |
| --- | --- | --- | --- |
| SPEC-b08457ae | ✅ | SHA256 hash function... | REQ-8e6243b6 |

## Implementation
Implemented modules: 1

### `src/crypto/hash_utils.py`
✅ def hash_data(data: bytes) -> str:

---
*Auto-generated by Paragon Documenter on 2025-12-06 06:08:52 UTC*
*Graph contains 3 nodes and 2 edges*
```

### Example Wiki Page

```markdown
# SPEC-b08457ae: SHA256 hash function

## Metadata
- **Status**: VERIFIED
- **Created**: 2025-12-06T06:08:52+00:00
- **Created By**: architect-agent

## Specification
SHA256 hash function: accept bytes input, return hex string

## Requirement
**REQ-8e6243b6**
Implement a cryptographic hash function module

## Implementation
### CODE-97b726ae
**File**: `src/crypto/hash_utils.py`
```python
def hash_data(data: bytes) -> str:
    ...
```
```

---

## 3. AuditLogger

### Purpose
Forensic audit logger for all graph mutations with complete traceability.

### Key Features
- **Structured Logging**: JSONL format (one JSON object per line)
- **Complete Context**: `{timestamp, agent_id, node_id, merkle_hash, action}`
- **Queryable**: Filter by time range, agent, node, action type
- **Thread-Safe**: Concurrent logging supported

### Storage Format

Each log entry is a single line of JSON:

```json
{
  "timestamp": "2025-12-06T06:08:52.201095+00:00",
  "agent_id": "builder-gpt4-v1",
  "agent_role": "BUILDER",
  "node_id": "97b726ae0f4f4e6aa637953297adbe9a",
  "merkle_hash": null,
  "action": "node_created",
  "details": {
    "node_type": "CODE",
    "file_path": "src/crypto/hash_utils.py"
  }
}
```

### Usage

```python
from infrastructure.logger import AuditLogger, log_audit

# Initialize
audit = AuditLogger(log_path=Path("data/audit.log"))

# Log an action
audit.log_action(
    action="node_created",
    node_id="node_123",
    agent_id="builder-gpt4-v1",
    agent_role="BUILDER",
    merkle_hash="abc123def456",
    file_path="/src/hash.py"  # Additional details
)

# Or use convenience function
log_audit(
    action="status_changed",
    node_id="node_123",
    agent_id="tester-agent",
    old_status="PENDING",
    new_status="VERIFIED"
)
```

### Queries

```python
# Get node history
history = audit.get_node_history("node_123")

# Get agent activity
activity = audit.get_agent_activity("builder-gpt4-v1")

# Get recent entries
recent = audit.get_recent(n=100)

# Filter by criteria
entries = audit.read_entries(
    since="2025-12-06T00:00:00Z",
    agent_id="builder-gpt4-v1",
    action="node_created"
)
```

---

## Integration Pattern

The Historian module follows this pattern for each graph transaction:

```python
# 1. Create graph transaction
nodes_created = []
edges_created = []

req = NodeData.create(type="REQ", content="...")
db.add_node(req)
nodes_created.append(req.id)

spec = NodeData.create(type="SPEC", content="...")
db.add_node(spec)
nodes_created.append(spec.id)

edge = EdgeData.create(source=spec.id, target=req.id, type="TRACES_TO")
db.add_edge(edge)
edges_created.append((spec.id, req.id, "TRACES_TO"))

# 2. Log to audit trail
log_audit(
    action="node_created",
    node_id=req.id,
    agent_id="user",
    merkle_hash=req.merkle_hash
)

# 3. Trigger git commit on transaction boundary
git_sync.on_transaction_complete(
    nodes_created=nodes_created,
    edges_created=edges_created,
    agent_id="builder-gpt4-v1"
)

# 4. Generate documentation (on major cycles)
documenter.generate_readme()
documenter.generate_wiki()

# 5. Update changelog
documenter.append_changelog(
    old_merkle=previous_merkle,
    new_merkle=current_merkle,
    description="Feature implementation"
)
```

---

## File Structure

```
paragon/
├── infrastructure/
│   ├── git_sync.py          # GitSync module
│   └── logger.py            # MutationLogger + AuditLogger
├── agents/
│   └── documenter.py        # Documenter agent
├── config/
│   └── paragon.toml         # Configuration
├── data/
│   └── audit.log            # Audit trail (JSONL)
├── docs/
│   └── wiki/                # Auto-generated wiki pages
│       └── spec_*.md
├── README.md                # Auto-generated README
└── CHANGELOG.md             # Auto-generated changelog
```

---

## Testing

Run the test suite:

```bash
python workspace/test_historian.py
```

Run the integration example:

```bash
python workspace/historian_example.py
```

---

## Design Philosophy

### Graph as Source of Truth
Documentation is generated from the graph state, not from prompts. This ensures:
- No drift between docs and implementation
- Docs always reflect reality
- Updates are automatic on graph mutations

### Forensic Traceability
Every action is logged with complete context:
- **Who**: agent_id and agent_role
- **What**: action and node_id
- **When**: ISO8601 timestamp
- **State**: merkle_hash of affected node/graph

This enables:
- Time-travel debugging
- Blame analysis
- Compliance auditing
- Rollback to previous states

### Semantic Versioning via Teleology
Git commits derive meaning from the graph structure itself:
- CODE nodes → `feat:` commits
- SPEC nodes → `docs:` commits
- REQ nodes → `chore:` commits

The teleology chain (CODE → SPEC → REQ) provides context for commit messages, making the Git history a semantic representation of feature development.

---

## Future Enhancements

1. **Differential Wiki**: Only regenerate changed SPEC pages
2. **Merkle Tree Visualization**: Visual diff of graph state between commits
3. **Compliance Reports**: Auto-generate audit reports for governance
4. **Git Blame Integration**: Link code lines to originating REQ/SPEC nodes
5. **Rollback Support**: Restore graph state from git commit

---

## Dependencies

No additional dependencies beyond core Paragon:
- `msgspec` - Data structures
- `polars` - For potential CSV export of audit logs
- `git` - System binary for git operations

---

## License

Part of Project Paragon - see main LICENSE file.
