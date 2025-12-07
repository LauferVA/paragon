# PARAGON COMPREHENSIVE TEST PLAN

## Executive Summary

After thorough investigation of the Paragon codebase, I've identified significant testing gaps across all layers. The project has 46 Python files with only 5 protocol benchmarks and 2 ad-hoc workspace tests. A structured test suite is critical for production readiness.

---

## Current Test Coverage Analysis

### Existing Tests
**Location**: `/Users/lauferva/paragon/benchmarks/`

1. **Speed Tests** (benchmarks/protocol_alpha.py)
   - Wave computation (rustworkx layers)
   - Batch node insertion
   - Descendant queries
   - Adjacency matrix export
   - Cycle detection
   - Status: PASSING (benchmarks only, not assertions)

2. **Integrity Tests** (benchmarks/protocol_beta.py)
   - Self-ingestion (code parsing)
   - Import tracking
   - Index map consistency
   - Duplicate handling
   - Serialization round-trip
   - Wave correctness
   - Status: PASSING (integration-level)

3. **Orchestration Tests** (tests/integration/test_orchestrator.py)
   - Tool functions (add_node, batch ops, queries)
   - StateGraph creation
   - Human loop controller
   - Integration tests
   - Status: PASSING (mocked LLM)

4. **Brain/LLM Tests** (tests/unit/agents/test_tools_llm.py)
   - Syntax checking (tree-sitter)
   - add_node_safe hooks
   - Alignment verification
   - Schema validation (msgspec)
   - Prompt builders
   - Status: PASSING (unittest-based)

5. **Physics/Invariant Tests** (tests/unit/core/test_graph_invariants.py)
   - Graph invariants (Handshaking, Balis, DAG)
   - Teleology validation
   - Merkle hashing
   - Full workflow integration
   - Status: PASSING (unittest-based)

**Ad-hoc Tests**:
- `/Users/lauferva/paragon/workspace/test_rerun_logger.py` - Manual RerunLogger test
- `/Users/lauferva/paragon/workspace/test_historian.py` - Manual Historian test

**Empty Test Directory**: `/Users/lauferva/paragon/tests/` (exists but empty)

---

## Testing Gaps by Category

### 1. UNIT TESTS (CRITICAL Priority)

#### Core Layer (`/Users/lauferva/paragon/core/`)

**graph_db.py** - MISSING ALL UNIT TESTS
```
Critical test cases:
- test_add_node_duplicate_rejection
- test_get_node_not_found_error
- test_batch_operations_atomicity
- test_edge_creation_node_validation
- test_get_waves_empty_graph
- test_get_descendants_circular_reference_protection
- test_update_node_content_merkle_rehash
- test_delete_node_cascade_edges
- test_node_map_inv_map_bidirectional_consistency
- test_query_by_type_filtering
- test_query_by_status_filtering
- test_export_to_dataframe_polars
- test_get_graph_stats_accuracy
- test_concurrent_access_thread_safety (if needed)
```

**llm.py** - MISSING CRITICAL TESTS
```
Critical test cases:
- test_rate_limit_guard_throttling
- test_rate_limit_guard_retry_after_429
- test_model_router_high_reasoning_task
- test_model_router_mundane_task
- test_model_router_sensitive_task_local_only
- test_structured_llm_json_schema_generation
- test_structured_llm_retry_on_validation_error
- test_structured_llm_max_retries_exceeded
- test_litellm_timeout_handling
- test_token_counting_accuracy
- test_schema_prompt_injection_resistance
```

**ontology.py** - PARTIALLY COVERED (Physics/Invariant Tests)
```
Additional test cases needed:
- test_topology_constraint_validation_all_types
- test_edge_constraint_mode_enforcement
- test_structural_trigger_pattern_matching
- test_get_required_edges_for_node_type
- test_validate_status_for_type_transitions
```

**schemas.py** - PARTIALLY COVERED (Integrity Tests)
```
Additional test cases needed:
- test_nodedata_validation_required_fields
- test_edgedata_source_target_equality_rejection
- test_compute_hash_collision_resistance
- test_merkle_hash_dependency_ordering
- test_serialization_performance_benchmark
```

**teleology.py** - COVERED (Physics/Invariant Tests) ✓

**graph_invariants.py** - COVERED (Physics/Invariant Tests) ✓

**alignment.py** - MISSING ALL TESTS
```
Critical test cases:
- test_feature_extraction_node_embeddings
- test_affinity_matrix_construction
- test_rrwm_algorithm_convergence
- test_hungarian_discrete_matching
- test_alignment_score_threshold_calibration
- test_align_graphs_empty_graph_handling
- test_align_graphs_single_node_matching
- test_performance_large_graphs (>1000 nodes)
```

**token_counter.py** - MISSING ALL TESTS
```
Critical test cases:
- test_count_tokens_various_encodings
- test_count_tokens_special_characters
- test_count_tokens_unicode
- test_estimate_cost_gpt4
- test_estimate_cost_claude
```

**resource_guard.py** - MISSING ALL TESTS
```
Critical test cases:
- test_resource_monitor_thread_lifecycle
- test_pause_signal_on_high_memory
- test_pause_signal_on_high_cpu
- test_resume_signal_on_recovery
- test_wait_for_resources_blocking
- test_stop_monitoring_cleanup
```

#### Agents Layer (`/Users/lauferva/paragon/agents/`)

**tools.py** - PARTIALLY COVERED (Orchestration Tests)
```
Additional test cases needed:
- test_parse_source_invalid_syntax_handling
- test_parse_source_unsupported_language
- test_align_node_sets_threshold_validation
- test_check_syntax_all_error_types
- test_verify_alignment_missing_nodes
- test_safe_node_result_violation_reporting
```

**orchestrator.py** - PARTIALLY COVERED (Orchestration Tests)
```
Additional test cases needed:
- test_tdd_cycle_full_success_path
- test_tdd_cycle_test_failure_retry
- test_tdd_cycle_max_iterations_exceeded
- test_dialectic_phase_ambiguity_detection
- test_research_phase_web_search_integration
- test_human_checkpoint_pause_resume
- test_checkpointing_sqlite_persistence
- test_state_merge_annotated_reducers
- test_diagnostics_integration
- test_graceful_degradation_llm_unavailable
```

**human_loop.py** - PARTIALLY COVERED (Orchestration Tests)
```
Additional test cases needed:
- test_create_request_timeout_expiration
- test_submit_response_validation_selection_options
- test_cancel_request_status_update
- test_get_session_requests_filtering
- test_transition_matrix_invalid_transition_rejection
- test_pause_point_registry_custom_points
```

**schemas.py** - PARTIALLY COVERED (Brain/LLM Tests)
```
Additional test cases needed:
- test_implementation_plan_dependency_cycle_detection
- test_code_generation_imports_validation
- test_test_generation_assertions_presence
- test_dialector_output_ambiguity_scoring
- test_research_artifact_confidence_calculation
```

**prompts.py** - PARTIALLY COVERED (Brain/LLM Tests)
```
Additional test cases needed:
- test_build_architect_prompt_context_injection
- test_build_builder_prompt_spec_content_inclusion
- test_build_dialector_prompt_requirements_parsing
- test_build_researcher_prompt_query_formulation
- test_prompt_injection_sanitization
```

**research.py** - MISSING ALL TESTS
```
Critical test cases:
- test_research_state_machine_transitions
- test_socratic_loop_termination_conditions
- test_web_search_result_processing
- test_research_artifact_node_creation
- test_critique_phase_quality_assessment
- test_synthesize_phase_finding_aggregation
- test_max_iterations_limit
```

**documenter.py** - MISSING ALL TESTS
```
Critical test cases:
- test_generate_readme_content_structure
- test_generate_wiki_verified_nodes_only
- test_append_changelog_merkle_tracking
- test_auto_generate_on_graph_change
- test_include_pending_nodes_flag
- test_markdown_escaping
```

**tools_web.py** - MISSING ALL TESTS
```
Critical test cases:
- test_search_web_tavily_integration
- test_search_result_parsing
- test_search_response_error_handling
- test_rate_limiting_web_requests
- test_search_timeout_handling
```

#### Infrastructure Layer (`/Users/lauferva/paragon/infrastructure/`)

**git_sync.py** - PARTIALLY COVERED (workspace test)
```
Additional test cases needed:
- test_on_transaction_complete_commit_creation
- test_semantic_commit_message_generation
- test_teleology_chain_traversal_req_spec_code
- test_agent_attribution_in_commit
- test_auto_push_disabled_by_default
- test_git_command_error_handling
- test_repo_path_validation
```

**rerun_logger.py** - PARTIALLY COVERED (workspace test)
```
Additional test cases needed:
- test_log_node_position_3d_space
- test_log_edge_arrow_visualization
- test_log_thought_timeline_sequence
- test_log_code_diff_syntax_highlighting
- test_log_graph_snapshot_full_state
- test_session_id_file_naming
- test_rrd_file_creation_and_cleanup
```

**logger.py** - MISSING ALL TESTS
```
Critical test cases:
- test_audit_logger_entry_format
- test_audit_logger_file_rotation
- test_log_node_created_action
- test_log_status_changed_action
- test_get_node_history_filtering
- test_read_entries_pagination
- test_correlation_id_tracking
```

**diagnostics.py** - MISSING ALL TESTS
```
Critical test cases:
- test_state_summary_global_db_status
- test_state_summary_llm_status
- test_llm_call_metric_tracking
- test_phase_timing_measurement
- test_correlation_id_generation
- test_print_summary_output_format
- test_bloat_detection_thresholds
- test_reset_diagnostics_cleanup
```

#### Domain Layer (`/Users/lauferva/paragon/domain/`)

**code_parser.py** - PARTIALLY COVERED (Integrity Tests self-ingestion)
```
Additional test cases needed:
- test_parse_classes_extraction
- test_parse_functions_extraction
- test_parse_imports_extraction
- test_parse_invalid_syntax_fault_tolerance
- test_parse_incremental_reparse
- test_parse_multi_language_support
- test_query_cursor_captures_accuracy
- test_s_expression_query_syntax_errors
```

#### Requirements Layer (`/Users/lauferva/paragon/requirements/`)

**socratic_engine.py** - MISSING ALL TESTS
```
Critical test cases:
- test_gap_analyzer_missing_information_detection
- test_canonical_question_level_hierarchy
- test_socratic_engine_question_flow
- test_terminal_edge_out_of_scope_requirement
- test_structural_validators_conditional_triggering
- test_minimal_sufficient_statistic_optimization
- test_question_category_routing
```

---

### 2. COMPONENT TESTS (HIGH Priority)

**LLM Wrapper with Mocked Responses**
```python
# File: tests/component/test_llm_component.py
- test_structured_llm_generate_with_mock_openai
- test_structured_llm_generate_with_mock_anthropic
- test_rate_limit_guard_integration
- test_model_router_task_dispatch
- test_retry_logic_transient_failures
- test_context_window_overflow_handling
```

**Graph Operations (ParagonDB)**
```python
# File: tests/component/test_graph_component.py
- test_add_node_with_topology_validation
- test_add_edge_with_constraint_checking
- test_batch_operations_transaction_semantics
- test_wave_computation_complex_dag
- test_teleology_validation_on_insert
- test_merkle_rehash_on_content_change
```

**Human Loop Controller**
```python
# File: tests/component/test_human_loop_component.py
- test_request_lifecycle_pending_to_responded
- test_timeout_handling_async
- test_selection_validation_options
- test_approval_binary_response
- test_feedback_freeform_collection
- test_session_isolation
```

**Code Parser with Real Files**
```python
# File: tests/component/test_code_parser_component.py
- test_parse_real_python_file
- test_parse_directory_recursive
- test_extract_imports_from_project
- test_build_dependency_graph_from_codebase
- test_fault_tolerance_broken_files
```

**Git Sync with Real Repository**
```python
# File: tests/component/test_git_sync_component.py
- test_commit_creation_in_test_repo
- test_semantic_message_from_teleology
- test_auto_commit_on_transaction
- test_agent_attribution_coauthor
- test_no_commit_on_disabled
```

---

### 3. INTEGRATION TESTS (HIGH Priority)

**Orchestrator + Tools + Graph**
```python
# File: tests/integration/test_orchestrator_integration.py
- test_full_tdd_cycle_init_to_passed
- test_tdd_cycle_with_test_failures_retry_loop
- test_dialectic_research_plan_build_flow
- test_human_checkpoint_interruption_resume
- test_checkpointing_persistence_across_restarts
- test_diagnostics_tracking_throughout_cycle
```

**LLM + Graph Persistence**
```python
# File: tests/integration/test_llm_graph_integration.py
- test_architect_generates_plan_creates_spec_nodes
- test_builder_generates_code_creates_code_nodes
- test_alignment_verification_spec_code_matching
- test_syntax_validation_rejects_invalid_code
- test_merkle_hash_propagation_through_chain
```

**Human Loop + Orchestrator State Machine**
```python
# File: tests/integration/test_human_loop_integration.py
- test_pause_on_approval_required
- test_resume_after_human_response
- test_timeout_fallback_to_default
- test_escalation_on_repeated_failures
- test_selection_integration_with_state
```

**Code Parser + Graph Ingestion**
```python
# File: tests/integration/test_parser_graph_integration.py
- test_parse_codebase_ingest_to_graph
- test_import_edges_creation
- test_class_function_hierarchy
- test_teleology_validation_after_ingestion
```

**Git Sync + Documenter + Graph Changes**
```python
# File: tests/integration/test_historian_integration.py
- test_graph_change_triggers_commit_and_docs
- test_changelog_append_on_merkle_change
- test_readme_regeneration_on_new_nodes
- test_wiki_update_on_spec_changes
```

---

### 4. END-TO-END TESTS (MEDIUM Priority)

**Full TDD Cycle with Mocked LLM**
```python
# File: tests/e2e/test_tdd_cycle_e2e.py
- test_complete_feature_implementation
  - User submits requirement
  - Dialectic identifies ambiguities
  - Human clarifies scope
  - Research gathers context
  - Architect creates plan
  - Builder generates code
  - Tests verify correctness
  - Documentation auto-generated
  - Git commit created
```

**Dialectic → Clarification → Research → Plan → Build → Test Flow**
```python
# File: tests/e2e/test_full_workflow_e2e.py
- test_ambiguous_requirement_clarification_loop
- test_research_findings_inform_architecture
- test_plan_approval_human_checkpoint
- test_build_phase_code_generation
- test_test_phase_verification
- test_fix_phase_retry_on_failure
```

**Human-in-the-Loop Interruption and Resumption**
```python
# File: tests/e2e/test_human_interruption_e2e.py
- test_pause_at_plan_approval
- test_resume_after_approval
- test_pause_at_code_review
- test_resume_with_feedback
- test_checkpoint_persistence_across_sessions
```

---

### 5. SECURITY TESTS (CRITICAL Priority)

**Code Injection in Generated Code**
```python
# File: tests/security/test_code_injection_security.py
- test_tree_sitter_rejects_malicious_code
- test_syntax_validation_blocks_eval_exec
- test_prompt_injection_sanitization
- test_import_statement_whitelist
- test_filesystem_access_restrictions
```

**Prompt Injection Resistance**
```python
# File: tests/security/test_prompt_injection_security.py
- test_system_prompt_override_prevention
- test_context_escape_detection
- test_json_injection_in_structured_output
- test_delimiter_confusion_attack
```

**Rate Limiting and Resource Guards**
```python
# File: tests/security/test_resource_security.py
- test_rate_limit_guard_blocks_excessive_requests
- test_resource_guard_prevents_oom
- test_cpu_throttling_on_high_load
- test_concurrent_request_limiting
```

---

## Test Infrastructure Structure

```
/Users/lauferva/paragon/tests/
├── unit/
│   ├── core/
│   │   ├── test_graph_db.py
│   │   ├── test_llm.py
│   │   ├── test_ontology.py
│   │   ├── test_schemas.py
│   │   ├── test_alignment.py
│   │   ├── test_token_counter.py
│   │   └── test_resource_guard.py
│   ├── agents/
│   │   ├── test_tools.py
│   │   ├── test_orchestrator.py
│   │   ├── test_human_loop.py
│   │   ├── test_schemas.py
│   │   ├── test_prompts.py
│   │   ├── test_research.py
│   │   ├── test_documenter.py
│   │   └── test_tools_web.py
│   ├── infrastructure/
│   │   ├── test_git_sync.py
│   │   ├── test_rerun_logger.py
│   │   ├── test_logger.py
│   │   ├── test_diagnostics.py
│   │   └── test_metrics.py
│   ├── domain/
│   │   └── test_code_parser.py
│   └── requirements/
│       └── test_socratic_engine.py
├── component/
│   ├── test_llm_component.py
│   ├── test_graph_component.py
│   ├── test_human_loop_component.py
│   ├── test_code_parser_component.py
│   └── test_git_sync_component.py
├── integration/
│   ├── test_orchestrator_integration.py
│   ├── test_llm_graph_integration.py
│   ├── test_human_loop_integration.py
│   ├── test_parser_graph_integration.py
│   └── test_historian_integration.py
├── e2e/
│   ├── test_tdd_cycle_e2e.py
│   ├── test_full_workflow_e2e.py
│   ├── test_human_interruption_e2e.py
│   └── test_multi_session_e2e.py
├── security/
│   ├── test_code_injection_security.py
│   ├── test_prompt_injection_security.py
│   ├── test_resource_security.py
│   └── test_data_sanitization_security.py
├── fixtures/
│   ├── sample_code.py
│   ├── malformed_code.py
│   ├── sample_requirements.json
│   └── mock_llm_responses.json
└── conftest.py  # pytest configuration and shared fixtures
```

---

## Prioritized Implementation Plan

### Phase 1: Critical Unit Tests (Week 1)
**Priority**: CRITICAL
**Files**: 15 test files
**Focus**: Core graph operations, LLM wrapper, safety hooks
1. `tests/unit/core/test_graph_db.py` (20 tests)
2. `tests/unit/core/test_llm.py` (10 tests)
3. `tests/unit/agents/test_tools.py` (15 tests)
4. `tests/unit/agents/test_orchestrator.py` (10 tests)
5. `tests/security/test_code_injection_security.py` (8 tests)

**Exit Criteria**: 80% unit test coverage for core/ and agents/tools.py

### Phase 2: Component & Integration Tests (Week 2)
**Priority**: HIGH
**Files**: 10 test files
**Focus**: Component interactions, state management
1. `tests/component/test_llm_component.py`
2. `tests/component/test_graph_component.py`
3. `tests/integration/test_orchestrator_integration.py`
4. `tests/integration/test_llm_graph_integration.py`
5. `tests/security/test_prompt_injection_security.py`

**Exit Criteria**: All critical paths through orchestrator tested

### Phase 3: E2E & Security Tests (Week 3)
**Priority**: HIGH
**Files**: 8 test files
**Focus**: Full workflows, security hardening
1. `tests/e2e/test_tdd_cycle_e2e.py`
2. `tests/e2e/test_full_workflow_e2e.py`
3. `tests/security/test_resource_security.py`
4. `tests/security/test_data_sanitization_security.py`

**Exit Criteria**: At least 2 complete E2E scenarios passing

### Phase 4: Extended Coverage (Week 4)
**Priority**: MEDIUM
**Files**: Remaining 20 test files
**Focus**: Edge cases, infrastructure, domain layers
1. Infrastructure tests (git_sync, rerun_logger, diagnostics)
2. Domain tests (code_parser, alignment)
3. Requirements tests (socratic_engine)
4. API tests (routes)

**Exit Criteria**: 90% overall code coverage

---

## Protocol Zeta: Human-in-the-Loop Testing

**NEW PROTOCOL NEEDED** for testing DIALECTIC, CLARIFICATION, and RESEARCH phases.

### Protocol Zeta Structure

**1. Schema Validation Tests**
```python
test_dialector_output_schema()      # Required fields: is_clear, ambiguities, blocking_count
test_research_artifact_schema()     # Required fields: task_category, input_contract, output_contract
test_ambiguity_marker_schema()      # Impact classification: BLOCKING vs CLARIFYING
```

**2. Phase Transition Tests**
```python
test_dialectic_to_clarification()   # Ambiguities found -> wait for human
test_dialectic_to_research()        # No ambiguities -> proceed
test_clarification_to_research()    # Human responded -> proceed
test_clarification_to_dialectic()   # Re-check after clarification
```

**3. MockHumanLoopController**
```python
class MockHumanLoopController(HumanLoopController):
    """Controller that auto-responds to requests with pre-defined answers."""

    def __init__(self, responses: Dict[str, str] = None):
        super().__init__()
        self.responses = responses or {}
        self.request_history = []

    def create_request(self, pause_point, session_id, prompt, context=None, priority=None):
        request = super().create_request(pause_point, session_id, prompt, context, priority)
        self.request_history.append(request)

        if pause_point.id in self.responses:
            self.submit_response(request.id, self.responses[pause_point.id])

        return request
```

---

## Test Metrics & Success Criteria

### Coverage Targets
- **Unit Tests**: 90% line coverage for core/, agents/
- **Integration Tests**: 80% coverage for workflows
- **E2E Tests**: All critical user journeys covered
- **Security Tests**: 100% coverage for injection points

### Quality Gates
1. **All tests must pass** before merge
2. **No flaky tests** (>99% success rate over 10 runs)
3. **Performance regression** <5% vs baseline (Protocol Alpha)
4. **Security tests** must all pass (zero tolerance)

---

## Conclusion

**Current State**: 5 protocol benchmarks, 2 ad-hoc tests, empty test directory
**Required State**: 70+ test files, 500+ test cases, 90% coverage

**Estimated Effort**: 4 weeks (1 developer)
**Risk**: Production deployment without comprehensive tests is HIGH RISK

**Recommendation**: Prioritize Phase 1 (Critical Unit Tests) and Security Tests before any production release.
