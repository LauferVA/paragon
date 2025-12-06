"""
Unit tests for learning system schemas (Branch A: Schema Foundation).

Tests the foundational schemas that enable the learning system:
- AgentSignature: Immutable audit trail for agent actions
- SignatureChain: Evolution tracking for nodes
- SignatureAction, CyclePhase, FailureCode, NodeOutcome enums
- Integration with add_node_safe for signature storage
"""
import pytest
import msgspec
from datetime import datetime
import uuid

from agents.schemas import (
    AgentSignature,
    SignatureChain,
    SignatureAction,
    CyclePhase,
    FailureCode,
    NodeOutcome,
)
from agents.tools import add_node_safe, get_db
from core.ontology import NodeType


# =============================================================================
# ENUM TESTS
# =============================================================================

def test_signature_action_enum():
    """
    Validate that SignatureAction enum has all required values.

    Verifies:
    - All expected actions are defined
    - Values are strings
    - Enum is properly typed
    """
    assert SignatureAction.CREATED.value == "created"
    assert SignatureAction.MODIFIED.value == "modified"
    assert SignatureAction.VERIFIED.value == "verified"
    assert SignatureAction.REJECTED.value == "rejected"
    assert SignatureAction.ESCALATED.value == "escalated"
    assert SignatureAction.SUPERSEDED.value == "superseded"

    # Verify it's a string enum
    assert isinstance(SignatureAction.CREATED.value, str)


def test_cycle_phase_enum():
    """
    Validate that CyclePhase enum has all orchestrator phases.

    Verifies:
    - All orchestrator phases are represented
    - Values match expected strings
    """
    assert CyclePhase.INIT.value == "init"
    assert CyclePhase.DIALECTIC.value == "dialectic"
    assert CyclePhase.RESEARCH.value == "research"
    assert CyclePhase.CLARIFICATION.value == "clarification"
    assert CyclePhase.PLAN.value == "plan"
    assert CyclePhase.BUILD.value == "build"
    assert CyclePhase.TEST.value == "test"
    assert CyclePhase.PASSED.value == "passed"
    assert CyclePhase.FAILED.value == "failed"


def test_failure_code_enum():
    """
    Validate that FailureCode enum categorizes failure modes.

    Verifies:
    - All failure categories are defined
    - Codes follow F1-F5 convention
    """
    assert FailureCode.F1.value == "F1"
    assert FailureCode.F2.value == "F2"
    assert FailureCode.F3.value == "F3"
    assert FailureCode.F4.value == "F4"
    assert FailureCode.F5.value == "F5"


def test_node_outcome_enum():
    """
    Validate that NodeOutcome enum has all outcome states.

    Verifies:
    - All possible outcomes are defined
    - Values are descriptive strings
    """
    assert NodeOutcome.VERIFIED_SUCCESS.value == "verified_success"
    assert NodeOutcome.VERIFIED_FAILURE.value == "verified_failure"
    assert NodeOutcome.TEST_PROD_DIVERGENCE.value == "test_prod_divergence"
    assert NodeOutcome.UNEXERCISED.value == "unexercised"
    assert NodeOutcome.INDETERMINATE.value == "indeterminate"


# =============================================================================
# AGENT SIGNATURE TESTS
# =============================================================================

def test_agent_signature_creation():
    """
    Validate that AgentSignature can be created with all fields.

    Verifies:
    - Signature is created successfully
    - All fields are stored correctly
    - Signature is frozen (immutable)
    """
    timestamp = datetime.utcnow().isoformat()
    signature = AgentSignature(
        agent_id="Builder-v2",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4000, "cost_limit": 0.50},
        timestamp=timestamp
    )

    assert signature.agent_id == "Builder-v2"
    assert signature.model_id == "claude-sonnet-4-5-20250929"
    assert signature.phase == CyclePhase.BUILD
    assert signature.action == SignatureAction.CREATED
    assert signature.temperature == 0.7
    assert signature.context_constraints["max_tokens"] == 4000
    assert signature.timestamp == timestamp


def test_agent_signature_immutable():
    """
    Validate that AgentSignature is immutable (frozen=True).

    Verifies:
    - Cannot modify signature after creation
    - Raises appropriate error on modification attempt
    """
    signature = AgentSignature(
        agent_id="Tester-v1",
        model_id="claude-haiku-4-5-20251001",
        phase=CyclePhase.TEST,
        action=SignatureAction.VERIFIED,
        temperature=0.0,
        context_constraints={},
        timestamp=datetime.utcnow().isoformat()
    )

    # Attempt to modify should raise error
    with pytest.raises(AttributeError):
        signature.agent_id = "Modified"


def test_agent_signature_msgspec_serialization():
    """
    Validate that AgentSignature can be serialized/deserialized with msgspec.

    Verifies:
    - Signature can be encoded to bytes
    - Signature can be decoded back to object
    - Data integrity is preserved
    """
    original = AgentSignature(
        agent_id="Architect-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.PLAN,
        action=SignatureAction.CREATED,
        temperature=0.5,
        context_constraints={"session_id": "test-123"},
        timestamp="2025-01-01T00:00:00Z"
    )

    # Encode to bytes
    encoded = msgspec.msgpack.encode(original)
    assert isinstance(encoded, bytes)

    # Decode back to object
    decoded = msgspec.msgpack.decode(encoded, type=AgentSignature)
    assert decoded.agent_id == original.agent_id
    assert decoded.model_id == original.model_id
    assert decoded.phase == original.phase
    assert decoded.action == original.action


# =============================================================================
# SIGNATURE CHAIN TESTS
# =============================================================================

def test_signature_chain_creation():
    """
    Validate that SignatureChain can be created with single signature.

    Verifies:
    - Chain is created successfully
    - Initial signature is stored
    - Default values are correct
    """
    signature = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp=datetime.utcnow().isoformat()
    )

    chain = SignatureChain(
        node_id="CODE-abc123",
        state_id=str(uuid.uuid4()),
        signatures=[signature]
    )

    assert chain.node_id == "CODE-abc123"
    assert len(chain.signatures) == 1
    assert chain.signatures[0].agent_id == "Builder-v1"
    assert chain.is_replacement is False
    assert chain.replaced_node_id is None


def test_signature_chain_evolution():
    """
    Validate that SignatureChain tracks node evolution.

    When a node's content changes, we create a new state_id but keep
    the same node_id, and append a new signature.

    Verifies:
    - Same node_id across evolution
    - Different state_id for new version
    - Multiple signatures in chronological order
    """
    sig1 = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp="2025-01-01T00:00:00Z"
    )

    sig2 = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.MODIFIED,
        temperature=0.7,
        context_constraints={},
        timestamp="2025-01-01T01:00:00Z"
    )

    state_id_v1 = str(uuid.uuid4())
    state_id_v2 = str(uuid.uuid4())

    # Version 1 of the node
    chain_v1 = SignatureChain(
        node_id="CODE-xyz789",
        state_id=state_id_v1,
        signatures=[sig1]
    )

    # Version 2 of the same node (evolved)
    chain_v2 = SignatureChain(
        node_id="CODE-xyz789",  # Same node_id
        state_id=state_id_v2,    # Different state_id
        signatures=[sig1, sig2]  # Accumulated signatures
    )

    assert chain_v1.node_id == chain_v2.node_id
    assert chain_v1.state_id != chain_v2.state_id
    assert len(chain_v1.signatures) == 1
    assert len(chain_v2.signatures) == 2
    assert chain_v2.signatures[1].action == SignatureAction.MODIFIED


def test_signature_chain_replacement():
    """
    Validate that SignatureChain tracks node replacements.

    When a node is completely replaced by a new implementation,
    we create a new node_id but link back to the old one.

    Verifies:
    - Different node_id for replacement
    - is_replacement flag is True
    - replaced_node_id points to old node
    """
    original_sig = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp="2025-01-01T00:00:00Z"
    )

    replacement_sig = AgentSignature(
        agent_id="Builder-v2",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.SUPERSEDED,
        temperature=0.7,
        context_constraints={},
        timestamp="2025-01-02T00:00:00Z"
    )

    original_chain = SignatureChain(
        node_id="CODE-old123",
        state_id=str(uuid.uuid4()),
        signatures=[original_sig]
    )

    replacement_chain = SignatureChain(
        node_id="CODE-new456",     # New node_id
        state_id=str(uuid.uuid4()),
        signatures=[replacement_sig],
        is_replacement=True,
        replaced_node_id="CODE-old123"  # Links to original
    )

    assert replacement_chain.node_id != original_chain.node_id
    assert replacement_chain.is_replacement is True
    assert replacement_chain.replaced_node_id == original_chain.node_id


def test_signature_chain_msgspec_serialization():
    """
    Validate that SignatureChain can be serialized/deserialized.

    Verifies:
    - Chain with multiple signatures can be encoded
    - Decoded chain matches original
    - Nested signature objects are preserved
    """
    sig1 = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp="2025-01-01T00:00:00Z"
    )

    sig2 = AgentSignature(
        agent_id="Tester-v1",
        model_id="claude-haiku-4-5-20251001",
        phase=CyclePhase.TEST,
        action=SignatureAction.VERIFIED,
        temperature=0.0,
        context_constraints={},
        timestamp="2025-01-01T02:00:00Z"
    )

    original = SignatureChain(
        node_id="CODE-test",
        state_id=str(uuid.uuid4()),
        signatures=[sig1, sig2],
        is_replacement=False,
        replaced_node_id=None
    )

    # Encode and decode
    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=SignatureChain)

    assert decoded.node_id == original.node_id
    assert decoded.state_id == original.state_id
    assert len(decoded.signatures) == 2
    assert decoded.signatures[0].agent_id == "Builder-v1"
    assert decoded.signatures[1].agent_id == "Tester-v1"


# =============================================================================
# INTEGRATION WITH add_node_safe TESTS
# =============================================================================

def test_add_node_safe_without_signature(fresh_db):
    """
    Validate that add_node_safe works without signature (backward compatible).

    Verifies:
    - Function works as before when signature is not provided
    - Node is created successfully
    - No signature chain in metadata
    """
    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content="def test(): pass"
    )

    assert result.success is True
    assert result.node_id != ""

    # Verify no signature in metadata
    db = get_db()
    node = db.get_node(result.node_id)
    assert "_signature_chain" not in node.data


def test_add_node_safe_with_signature(fresh_db):
    """
    Validate that add_node_safe stores signature when provided.

    Verifies:
    - Signature is stored in node metadata
    - SignatureChain is created correctly
    - Node ID is set in the chain
    """
    signature = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={"max_tokens": 4000},
        timestamp=datetime.utcnow().isoformat()
    )

    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content="def example(): return 42",
        signature=signature
    )

    assert result.success is True
    assert result.node_id != ""

    # Verify signature chain is stored
    db = get_db()
    node = db.get_node(result.node_id)
    assert "_signature_chain" in node.data

    # Verify chain structure
    chain_dict = node.data["_signature_chain"]
    assert chain_dict["node_id"] == result.node_id
    assert len(chain_dict["signatures"]) == 1
    assert chain_dict["is_replacement"] is False
    assert chain_dict["replaced_node_id"] is None


def test_add_node_safe_signature_validation_failure(fresh_db):
    """
    Validate that signature is NOT stored if node creation fails.

    When syntax validation fails, the node should not be created,
    and therefore the signature should not be stored.

    Verifies:
    - Invalid code is rejected
    - Node count remains 0
    - No signature is persisted
    """
    signature = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp=datetime.utcnow().isoformat()
    )

    invalid_code = "def broken(: pass"

    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content=invalid_code,
        signature=signature
    )

    assert result.success is False
    assert result.syntax_valid is False

    # Verify no node was created
    db = get_db()
    assert db.node_count == 0


def test_add_node_safe_signature_metadata_preservation(fresh_db):
    """
    Validate that signature doesn't overwrite existing metadata.

    When a signature is provided along with other metadata,
    both should be preserved in the node.

    Verifies:
    - Custom metadata is preserved
    - Signature chain is added
    - Both coexist in node.data
    """
    signature = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp=datetime.utcnow().isoformat()
    )

    custom_metadata = {
        "filename": "example.py",
        "line_number": 42,
        "author": "test_user"
    }

    result = add_node_safe(
        node_type=NodeType.CODE.value,
        content="def example(): pass",
        data=custom_metadata,
        signature=signature
    )

    assert result.success is True

    # Verify both metadata and signature are stored
    db = get_db()
    node = db.get_node(result.node_id)

    # Custom metadata should be preserved
    assert node.data["filename"] == "example.py"
    assert node.data["line_number"] == 42
    assert node.data["author"] == "test_user"

    # Signature chain should also be present
    assert "_signature_chain" in node.data
    assert node.data["_signature_chain"]["node_id"] == result.node_id


# =============================================================================
# SIGNATURE CHAIN HELPER TESTS
# =============================================================================

def test_signature_chain_to_dict_conversion():
    """
    Validate that SignatureChain can be converted to dict for storage.

    Verifies:
    - msgspec.structs.asdict works correctly
    - All fields are preserved
    - Nested signatures are accessible
    """
    signature = AgentSignature(
        agent_id="Builder-v1",
        model_id="claude-sonnet-4-5-20250929",
        phase=CyclePhase.BUILD,
        action=SignatureAction.CREATED,
        temperature=0.7,
        context_constraints={},
        timestamp=datetime.utcnow().isoformat()
    )

    chain = SignatureChain(
        node_id="CODE-test",
        state_id=str(uuid.uuid4()),
        signatures=[signature]
    )

    chain_dict = msgspec.structs.asdict(chain)

    assert isinstance(chain_dict, dict)
    assert chain_dict["node_id"] == "CODE-test"
    assert len(chain_dict["signatures"]) == 1
    # Note: msgspec.structs.asdict doesn't recursively convert nested structs
    # The signature remains as an AgentSignature object
    assert isinstance(chain_dict["signatures"][0], AgentSignature)
    assert chain_dict["signatures"][0].agent_id == "Builder-v1"


def test_multiple_signatures_in_chain():
    """
    Validate that SignatureChain can hold multiple signatures.

    Simulates a node going through multiple agent interactions.

    Verifies:
    - Multiple signatures can be added
    - Order is preserved
    - Each signature is independent
    """
    sigs = [
        AgentSignature(
            agent_id="Dialector-v1",
            model_id="claude-haiku-4-5-20251001",
            phase=CyclePhase.DIALECTIC,
            action=SignatureAction.CREATED,
            temperature=0.3,
            context_constraints={},
            timestamp="2025-01-01T00:00:00Z"
        ),
        AgentSignature(
            agent_id="Researcher-v1",
            model_id="claude-sonnet-4-5-20250929",
            phase=CyclePhase.RESEARCH,
            action=SignatureAction.MODIFIED,
            temperature=0.5,
            context_constraints={},
            timestamp="2025-01-01T01:00:00Z"
        ),
        AgentSignature(
            agent_id="Architect-v1",
            model_id="claude-sonnet-4-5-20250929",
            phase=CyclePhase.PLAN,
            action=SignatureAction.VERIFIED,
            temperature=0.5,
            context_constraints={},
            timestamp="2025-01-01T02:00:00Z"
        ),
    ]

    chain = SignatureChain(
        node_id="SPEC-multi",
        state_id=str(uuid.uuid4()),
        signatures=sigs
    )

    assert len(chain.signatures) == 3
    assert chain.signatures[0].phase == CyclePhase.DIALECTIC
    assert chain.signatures[1].phase == CyclePhase.RESEARCH
    assert chain.signatures[2].phase == CyclePhase.PLAN

    # Verify chronological order is preserved
    assert chain.signatures[0].timestamp < chain.signatures[1].timestamp
    assert chain.signatures[1].timestamp < chain.signatures[2].timestamp
