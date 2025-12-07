# AGENTS LAYER AUDIT REPORT

## EXECUTIVE SUMMARY

| File | Purpose | Status | Fitness | Recommendation |
|------|---------|--------|---------|----------------|
| **orchestrator.py** | LangGraph TDD state machine with dialectic/research phases | FULLY_WIRED | HIGH | KEEP |
| **dispatch.py** | Graph-native agent dispatch using structural triggers | ORPHAN | MEDIUM | REMOVE |
| **documenter.py** | Auto-generates README/Wiki/Changelog from graph state | ORPHAN | HIGH | WIRE_IN or REMOVE |
| **tools.py** | Layer 7B auditor: graph ops, parsing, add_node_safe | FULLY_WIRED | HIGH | KEEP |
| **tools_web.py** | Tavily web search for research phase | FULLY_WIRED | MEDIUM | KEEP |
| **prompts.py** | Prompt builders for all agent roles | FULLY_WIRED | HIGH | KEEP |
| **schemas.py** | msgspec Struct definitions for agent outputs | FULLY_WIRED | HIGH | KEEP |
| **human_loop.py** | Human-in-the-loop controller for pause points | FULLY_WIRED | HIGH | KEEP |
| **research.py** | Nested research orchestrator (RESEARCH→CRITIQUE→SYNTHESIZE) | FULLY_WIRED | HIGH | KEEP |
| **quality_gate.py** | Layer 7B quality enforcement | PARTIALLY_WIRED | HIGH | WIRE_IN |
| **adaptive_questioner.py** | Learns question patterns for clarification | PARTIALLY_WIRED | MEDIUM | WIRE_IN |

## WIRING STATUS

| Status | Count | Files |
|--------|-------|-------|
| FULLY_WIRED | 7 | orchestrator, tools, tools_web, prompts, schemas, human_loop, research |
| PARTIALLY_WIRED | 2 | quality_gate, adaptive_questioner |
| ORPHAN | 2 | dispatch, documenter |

## CRITICAL FINDINGS

1. **dispatch.py** - ZERO imports. Dead code from abandoned Wave 4 refactor.
2. **documenter.py** - ZERO imports. Clean implementation but never called.
3. **quality_gate.py** - Only imported in tests, not in production orchestration.
4. **adaptive_questioner.py** - Only imported in tests/examples, not in clarification loop.

## RECOMMENDATIONS

### HIGH PRIORITY: WIRE IN
1. **quality_gate.py**: Integrate into test_node() to enforce quality floor
2. **adaptive_questioner.py**: Integrate into clarification_node() to optimize questions

### MEDIUM PRIORITY: DECIDE
3. **documenter.py**: Call generate_all_docs() at end of passed_node() OR remove

### DEPRECATION
4. **dispatch.py**: REMOVE - phase routing works fine in orchestrator; this is dead code

## ARCHITECTURE ALIGNMENT

**68% connected** (8/11 modules fully or partially wired)

- Core orchestration is solid
- Quality enforcement and learning systems need integration
- Two orphaned modules need decision (remove or wire in)
