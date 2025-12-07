# PARAGON CODEBASE AUDIT SYNTHESIS

**Date:** 2025-12-06
**Scope:** 75+ Python files across 6 layers
**Goal:** Identify orphaned, partially-wired, and unused code

---

## EXECUTIVE SUMMARY

| Layer | Files | Fully Wired | Partial | Orphan/Unused |
|-------|-------|-------------|---------|---------------|
| Core | 10 | 5 | 4 | 1 |
| Agents | 11 | 7 | 2 | 2 |
| Infrastructure | 12 | 1 | 5 | 6 |
| Benchmarks/API/Viz | 10 | 9 | 1 | 0 |
| Domain/Scripts | 5 | 2 | 0 | 3 |
| Examples/Workspace | 7 | 0 | 0 | 7 (intentional) |
| **TOTAL** | **55** | **24** | **12** | **19** |

**Overall Wiring Rate:** 44% fully wired, 22% partial, 34% orphan/unused

---

## FILES TO REMOVE (Dead Code)

| File | Reason | Lines Saved |
|------|--------|-------------|
| **agents/dispatch.py** | Wave 4 refactor abandoned; orchestrator uses LangGraph instead | 425 |
| **core/token_counter.py** | Zero imports anywhere; completely orphaned | ~200 |

**Total removable:** ~625 lines

---

## FILES TO WIRE IN (High-Quality Orphans)

### Priority 1: Critical (Blocks Learning/Observability)

| File | Gap | Integration Point |
|------|-----|-------------------|
| **infrastructure/git_sync.py** | No auto-commit on graph changes | Hook into add_node_safe() |
| **infrastructure/learning.py** | System never learns from failures | Hook into LLM model selection |
| **infrastructure/attribution.py** | Failures not analyzed | Hook into orchestrator error handlers |
| **infrastructure/divergence.py** | False positives undetected | Hook into test completion |
| **requirements/socratic_engine.py** | Dialectic engine orphaned | Wire into dialectic_node() |

### Priority 2: Important (Improves Quality)

| File | Gap | Integration Point |
|------|-----|-------------------|
| **core/analytics.py** | Only used ad-hoc | Add to orchestrator health checks |
| **core/teleology.py** | Golden thread not validated | Add to post-wave validation |
| **core/resource_guard.py** | OOM protection unused | Check signals in orchestrator loop |
| **agents/quality_gate.py** | Quality enforcement unused | Integrate into test_node() |
| **agents/documenter.py** | Auto-docs never generated | Call at end of passed_node() |

### Priority 3: Nice to Have

| File | Gap | Integration Point |
|------|-----|-------------------|
| **core/alignment.py** | Graph matching unused | Future refactoring detection |
| **infrastructure/diagnostics.py** | Correlation ID not linked | Link to mutation logger |
| **infrastructure/logger.py** | MutationLogger unused | Initialize in orchestrator |
| **agents/adaptive_questioner.py** | Learning not applied | Use in clarification_node() |

---

## WIRING STATUS BY FILE

### FULLY_WIRED (24 files) ✅
```
core/graph_db.py
core/ontology.py
core/schemas.py
core/llm.py
core/graph_invariants.py
agents/orchestrator.py
agents/tools.py
agents/tools_web.py
agents/prompts.py
agents/schemas.py
agents/human_loop.py
agents/research.py
infrastructure/training_store.py
benchmarks/harness.py
benchmarks/protocol_alpha.py  (Speed Tests)
benchmarks/protocol_beta.py   (Integrity Tests)
tests/integration/test_orchestrator.py  (Orchestration Tests - moved from protocol_gamma)
tests/unit/agents/test_tools_llm.py    (Brain/LLM Tests - moved from protocol_delta)
tests/unit/core/test_graph_invariants.py  (Physics Tests - moved from protocol_epsilon)
api/routes.py
domain/code_parser.py
main.py
```

### PARTIALLY_WIRED (12 files) ⚠️
```
core/analytics.py          → Used in ad-hoc queries only
core/alignment.py          → Only git_sync imports it
core/teleology.py          → Only quality_gate imports it
core/resource_guard.py     → Exported but never called
agents/quality_gate.py     → Only tests import it
agents/adaptive_questioner.py → Only tests/examples
infrastructure/config_graph.py → Self-referential only
infrastructure/rerun_logger.py → Benchmarks only
infrastructure/diagnostics.py → Lazy load, not fully used
infrastructure/environment.py → Standalone utility
infrastructure/logger.py    → Workspace examples only
viz/core.py                → API only, no direct tests
```

### ORPHAN/UNUSED (19 files) ❌
```
# Remove (dead code):
agents/dispatch.py         → 0 imports, abandoned refactor
core/token_counter.py      → 0 imports, never used

# Wire in (valuable):
infrastructure/git_sync.py  → 0 imports, ready to use
infrastructure/attribution.py → Demo only
infrastructure/divergence.py → Demo only
infrastructure/learning.py  → Demo only
infrastructure/data_loader.py → Future bulk import
infrastructure/metrics.py   → Future traceability
agents/documenter.py        → 0 imports, good implementation
requirements/socratic_engine.py → 0 imports, excellent design

# Standalone demos (correct):
run_dialectic.py
validate_api_fixes.py
examples/*.py (4 files)
workspace/*.py (3 files)
```

---

## RECOMMENDED ACTION PLAN

### Phase 1: Cleanup (Remove Dead Code)
```bash
git rm agents/dispatch.py
git rm core/token_counter.py
# Saves ~625 lines, reduces confusion
```

### Phase 2: Critical Wiring
1. Wire `git_sync.py` → auto-commit on graph changes
2. Wire `learning.py` → model selection learns from outcomes
3. Wire `attribution.py` → failures analyzed automatically
4. Wire `socratic_engine.py` → dialectic phase uses it

### Phase 3: Quality Wiring
5. Wire `quality_gate.py` → test phase enforces quality
6. Wire `documenter.py` → auto-generate docs on success
7. Wire `analytics.py` → health checks in orchestrator
8. Wire `resource_guard.py` → OOM protection active

### Phase 4: Nice-to-Have
9. Wire `teleology.py` → golden thread validation
10. Wire `adaptive_questioner.py` → smart question ordering
11. Link `diagnostics.py` → correlation IDs

---

## ARCHITECTURE HEALTH

**Strengths:**
- Core engine (graph_db, ontology, schemas) is solid
- Benchmarks layer is 100% wired
- API layer is production-ready
- All files follow CLAUDE.md directives (msgspec, rustworkx)

**Weaknesses:**
- Infrastructure layer is 50%+ orphaned
- Learning system exists but never invoked
- Quality enforcement exists but never called
- Documentation generation exists but never triggered

**Root Cause:**
Features were implemented but never integrated into the main orchestrator loop. The orchestrator is the hub, but spokes were built without connecting them.

---

## METRICS

| Metric | Value |
|--------|-------|
| Total Python files | 75+ |
| Audited files | 55 |
| Fully connected | 24 (44%) |
| Partially connected | 12 (22%) |
| Orphaned/Unused | 19 (34%) |
| Dead code (remove) | 2 files (~625 lines) |
| Valuable orphans (wire in) | 10 files |
| Standalone demos (keep) | 7 files |

---

## CONCLUSION

The Paragon codebase has **excellent architecture but incomplete integration**. The infrastructure layer contains high-quality, well-designed modules that were never wired into the orchestrator.

**Immediate actions:**
1. Remove 2 dead code files (dispatch.py, token_counter.py)
2. Wire 5 critical modules (git_sync, learning, attribution, divergence, socratic_engine)

This would bring wiring rate from 44% to ~60% and activate learning/observability features.
