# DOMAIN & SCRIPTS AUDIT REPORT

## EXECUTIVE SUMMARY

| File | Purpose | Status | Fitness | Recommendation |
|------|---------|--------|---------|----------------|
| **domain/code_parser.py** | Tree-sitter structural code parsing | FULLY_WIRED | HIGH | KEEP |
| **requirements/socratic_engine.py** | Dialectic specification engine | ORPHAN | MEDIUM | WIRE_IN |
| **main.py** | API server entry point (Granian ASGI) | FULLY_WIRED | HIGH | KEEP |
| **run_dialectic.py** | Interactive TDD orchestrator demo | UNUSED | MEDIUM | KEEP |
| **validate_api_fixes.py** | API validation/smoke test script | UNUSED | LOW | KEEP |

## WIRING STATUS

| Status | Count | Files |
|--------|-------|-------|
| FULLY_WIRED | 2 | code_parser, main |
| ORPHAN | 1 | socratic_engine |
| UNUSED (standalone) | 2 | run_dialectic, validate_api_fixes |

## CRITICAL FINDING

**socratic_engine.py** (756 lines) is a **high-quality orphan**:
- Excellent design matching CLAUDE.md philosophy
- Implements Socratic method for requirement clarification
- **ZERO imports** in entire codebase
- Designed for dialectic phase but **never wired into orchestrator**

## RECOMMENDATIONS

### Priority 1: WIRE_IN socratic_engine.py
**Effort:** 2-3 hours | **Impact:** High

Integration path:
1. Add SessionState to GraphState in orchestrator.py
2. Connect dialectic_node to SocraticEngine.create_session()
3. Connect clarification_node to record_answer()
4. Add unit tests

### Priority 2: INTEGRATE validate_api_fixes.py
Convert to pytest: tests/unit/api/test_api_validation.py

### Keep As-Is
- code_parser.py - Working perfectly
- main.py - Clean entry point
- run_dialectic.py - Valuable development tool
