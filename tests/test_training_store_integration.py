"""
Integration test for TrainingStore - Real-world usage demonstration.

This test demonstrates a complete learning workflow with multiple phases,
agent interactions, and outcome tracking.
"""
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from infrastructure import TrainingStore
from agents.schemas import (
    AgentSignature,
    SignatureChain,
    CyclePhase,
    FailureCode,
    NodeOutcome,
    SignatureAction,
)


def test_real_world_workflow():
    """
    Test a complete real-world learning workflow.

    Simulates:
    - A successful session with multiple phases and iterations
    - A failed session for comparison
    - Analysis queries to extract learnings
    """
    # Setup
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "integration_test.db"
    store = TrainingStore(db_path=db_path)

    try:
        # === Successful Session ===
        session_id = "session_20251206_001"

        # Phase 1: Dialectic - Create initial requirements
        dialectic_sig = AgentSignature(
            agent_id="dialectic_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.DIALECTIC,
            action=SignatureAction.CREATED,
            temperature=0.3,
            context_constraints={"max_tokens": 4096},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, dialectic_sig, "req_001", "state_001")

        # Phase 2: Plan - Architect creates plan
        plan_sig = AgentSignature(
            agent_id="architect_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.PLAN,
            action=SignatureAction.CREATED,
            temperature=0.5,
            context_constraints={"max_tokens": 8192},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, plan_sig, "plan_001", "state_002")

        # Phase 3: Build - Builder creates code (first attempt - rejected)
        build_sig_1 = AgentSignature(
            agent_id="builder_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.7,
            context_constraints={"max_tokens": 4096},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, build_sig_1, "code_001", "state_003")

        # Phase 4: Test - Tester rejects the code
        test_sig_1 = AgentSignature(
            agent_id="tester_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.TEST,
            action=SignatureAction.REJECTED,
            temperature=0.0,
            context_constraints={"max_tokens": 2048},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, test_sig_1, "code_001", "state_004")

        # Phase 5: Build - Builder modifies code
        build_sig_2 = AgentSignature(
            agent_id="builder_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.BUILD,
            action=SignatureAction.MODIFIED,
            temperature=0.7,
            context_constraints={"max_tokens": 4096},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, build_sig_2, "code_001", "state_005")

        # Phase 6: Test - Tester verifies the fix
        test_sig_2 = AgentSignature(
            agent_id="tester_agent",
            model_id="claude-sonnet-4-5",
            phase=CyclePhase.TEST,
            action=SignatureAction.VERIFIED,
            temperature=0.0,
            context_constraints={"max_tokens": 2048},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(session_id, test_sig_2, "code_001", "state_006")

        # Store complete signature chain for the code node
        chain = SignatureChain(
            node_id="code_001",
            state_id="state_006",
            signatures=[build_sig_1, test_sig_1, build_sig_2, test_sig_2],
            is_replacement=False,
            replaced_node_id=None,
        )
        store.record_signature_chain(chain)

        # Record successful session outcome
        store.record_session_outcome(
            session_id=session_id,
            outcome=NodeOutcome.VERIFIED_SUCCESS,
            stats={"total_nodes": 3, "total_iterations": 2, "total_tokens": 15360},
        )

        # === Failed Session ===
        failed_session_id = "session_20251206_002"

        failed_build_sig = AgentSignature(
            agent_id="builder_agent",
            model_id="claude-haiku-4",
            phase=CyclePhase.BUILD,
            action=SignatureAction.CREATED,
            temperature=0.8,
            context_constraints={"max_tokens": 2048},
            timestamp=datetime.utcnow().isoformat(),
        )
        store.record_attribution(
            failed_session_id, failed_build_sig, "code_002", "state_007"
        )

        store.record_session_outcome(
            session_id=failed_session_id,
            outcome=NodeOutcome.VERIFIED_FAILURE,
            failure_code=FailureCode.F2,
            failure_phase=CyclePhase.BUILD,
            stats={"total_nodes": 1, "total_iterations": 1, "total_tokens": 2048},
        )

        # === Analysis ===

        # Basic counts
        assert store.get_session_count() == 2
        assert store.get_attribution_count() == 7  # 6 from success + 1 from failure
        assert store.get_chain_count() == 1

        # Success rates by model
        sonnet_rate = store.get_success_rate("claude-sonnet-4-5", "build")
        haiku_rate = store.get_success_rate("claude-haiku-4", "build")

        # Sonnet: 1 success out of 1 session
        assert sonnet_rate == 1.0
        # Haiku: 0 successes out of 1 session
        assert haiku_rate == 0.0

        # Failure distribution
        failures = store.get_failure_distribution()
        assert failures == {"F2": 1}

        # Node history
        chain_retrieved = store.get_signature_chain("code_001")
        assert chain_retrieved is not None
        assert len(chain_retrieved.signatures) == 4
        assert [sig.action for sig in chain_retrieved.signatures] == [
            SignatureAction.CREATED,
            SignatureAction.REJECTED,
            SignatureAction.MODIFIED,
            SignatureAction.VERIFIED,
        ]

        # Session attributions
        attributions = store.get_attributions_by_session(session_id)
        assert len(attributions) == 6
        phases = [attr["phase"] for attr in attributions]
        assert "dialectic" in phases
        assert "plan" in phases
        assert "build" in phases
        assert "test" in phases

        # Verify session outcome details
        outcome = store.get_session_outcome(session_id)
        assert outcome["outcome"] == "verified_success"
        assert outcome["failure_code"] is None
        assert outcome["total_nodes"] == 3
        assert outcome["total_iterations"] == 2
        assert outcome["total_tokens"] == 15360

        failed_outcome = store.get_session_outcome(failed_session_id)
        assert failed_outcome["outcome"] == "verified_failure"
        assert failed_outcome["failure_code"] == "F2"
        assert failed_outcome["failure_phase"] == "build"

        print("\n=== Integration Test Results ===")
        print(f"Total sessions: {store.get_session_count()}")
        print(f"Total attributions: {store.get_attribution_count()}")
        print(f"Total chains: {store.get_chain_count()}")
        print(f"Success rate (claude-sonnet-4-5, BUILD): {sonnet_rate:.2%}")
        print(f"Success rate (claude-haiku-4, BUILD): {haiku_rate:.2%}")
        print(f"Failure distribution: {failures}")
        print(f"Node 'code_001' history: {len(chain_retrieved.signatures)} signatures")
        print("=== All Assertions Passed ===\n")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_real_world_workflow()
    print("Integration test completed successfully!")
