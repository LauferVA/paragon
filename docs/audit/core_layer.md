# PARAGON CORE LAYER AUDIT REPORT

## EXECUTIVE SUMMARY

| File | Purpose | Status | Fitness | Recommendation |
|------|---------|--------|---------|----------------|
| **graph_db.py** | Primary graph engine with rustworkx backend | FULLY_WIRED | EXCELLENT | KEEP |
| **ontology.py** | System vocabulary (NodeType, EdgeType enums) | FULLY_WIRED | EXCELLENT | KEEP |
| **schemas.py** | Data structures using msgspec | FULLY_WIRED | EXCELLENT | KEEP |
| **llm.py** | Structured LLM interface with rate limiting | FULLY_WIRED | EXCELLENT | KEEP |
| **analytics.py** | Read-only graph metrics | PARTIALLY_WIRED | GOOD | WIRE_IN |
| **alignment.py** | Graph matching via pygmtools | PARTIALLY_WIRED | GOOD | WIRE_IN |
| **graph_invariants.py** | Mathematical safety checks (DAG, Handshaking) | FULLY_WIRED | EXCELLENT | KEEP |
| **teleology.py** | Validates chain of causation to requirements | PARTIALLY_WIRED | GOOD | WIRE_IN |
| **resource_guard.py** | Background RAM/CPU monitor | PARTIALLY_WIRED | GOOD | WIRE_IN |
| **token_counter.py** | Model-aware token counting | UNUSED | GOOD | WIRE_IN or REMOVE |

## WIRING STATUS

| Status | Count | Files |
|--------|-------|-------|
| FULLY_WIRED | 5 | graph_db.py, ontology.py, schemas.py, llm.py, graph_invariants.py |
| PARTIALLY_WIRED | 4 | analytics.py, alignment.py, teleology.py, resource_guard.py |
| UNUSED | 1 | token_counter.py |

## CRITICAL FINDINGS

1. **token_counter.py** - ZERO imports anywhere in codebase. Completely orphaned.
2. **analytics.py** - Defined but only used in ad-hoc queries, not integrated into orchestrator
3. **alignment.py** - Graph matching ready but never called from main pipeline
4. **teleology.py** - "Golden Thread" validation exists but not actively run
5. **resource_guard.py** - Exported but no code calls init_resource_guard()

## RECOMMENDATIONS

### Priority 1: KEEP AS-IS (5 files)
- graph_db.py, ontology.py, schemas.py, llm.py, graph_invariants.py

### Priority 2: WIRE IN (4 files)
1. **analytics.py**: Add to orchestrator's post-wave health check
2. **alignment.py**: Integrate into refactoring/migration pipelines
3. **teleology.py**: Add to continuous integrity verification
4. **resource_guard.py**: Check signals before expensive operations

### Priority 3: RESOLVE (1 file)
1. **token_counter.py**: Either wire into llm.py or remove
