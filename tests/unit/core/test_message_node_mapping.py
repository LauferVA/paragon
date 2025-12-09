"""
Unit tests for message-to-node mapping infrastructure.

Tests the dialogue-to-graph correspondence features added for
supporting hover metadata and message-node bidirectional references.
"""
import pytest
from core.graph_db import ParagonDB, NodeNotFoundError
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, EdgeType, NodeStatus


class TestMessageNodeMapping:
    """Test suite for message-to-node mapping functionality."""

    def setup_method(self):
        """Set up a fresh database for each test."""
        self.db = ParagonDB()

    def test_link_message_to_node_references(self):
        """Test creating a REFERENCES edge from message to node."""
        # Create a MESSAGE node and a CODE node
        message = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Let's implement the authentication logic",
            data={"turn_number": 1, "session_id": "test_session"}
        )
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def authenticate(user): pass",
        )

        self.db.add_node(message)
        self.db.add_node(code)

        # Link message to code node
        edge_idx = self.db.link_message_to_node(
            message.id,
            code.id,
            edge_type=EdgeType.REFERENCES.value,
        )

        assert edge_idx >= 0
        assert self.db.has_edge(message.id, code.id)

        # Verify edge type
        edge = self.db.get_edge(message.id, code.id)
        assert edge.type == EdgeType.REFERENCES.value

    def test_link_message_to_node_defines_dialogue(self):
        """Test creating a DEFINES_DIALOGUE edge when message creates a node."""
        message = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Create a new SPEC for user authentication",
            data={"turn_number": 2}
        )
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="Specification for user authentication module",
        )

        self.db.add_node(message)
        self.db.add_node(spec)

        # Link with DEFINES_DIALOGUE edge
        edge_idx = self.db.link_message_to_node(
            message.id,
            spec.id,
            edge_type=EdgeType.DEFINES_DIALOGUE.value,
            metadata={"phase": "planning", "agent": "ARCHITECT"}
        )

        assert edge_idx >= 0
        edge = self.db.get_edge(message.id, spec.id)
        assert edge.type == EdgeType.DEFINES_DIALOGUE.value
        assert edge.metadata["phase"] == "planning"

    def test_get_messages_for_node(self):
        """Test retrieving all messages that reference a node."""
        # Create a CODE node
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def process_payment(): pass",
        )
        self.db.add_node(code)

        # Create multiple messages referencing this node
        message1 = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Review the payment processing code",
            data={"turn_number": 1}
        )
        message2 = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Add error handling to payment logic",
            data={"turn_number": 3}
        )
        message3 = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="This message doesn't reference the code node",
            data={"turn_number": 5}
        )

        self.db.add_node(message1)
        self.db.add_node(message2)
        self.db.add_node(message3)

        # Link messages to code
        self.db.link_message_to_node(message1.id, code.id)
        self.db.link_message_to_node(message2.id, code.id)

        # Get messages for code node
        messages = self.db.get_messages_for_node(code.id)

        assert len(messages) == 2
        message_ids = {m.id for m in messages}
        assert message1.id in message_ids
        assert message2.id in message_ids
        assert message3.id not in message_ids

    def test_get_messages_for_nonexistent_node(self):
        """Test get_messages_for_node with non-existent node."""
        messages = self.db.get_messages_for_node("nonexistent_id")
        assert messages == []

    def test_get_nodes_from_message(self):
        """Test retrieving all nodes referenced by a message."""
        # Create a message
        message = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Discuss authentication and payment modules",
            data={"turn_number": 4}
        )
        self.db.add_node(message)

        # Create multiple nodes
        spec1 = NodeData.create(
            type=NodeType.SPEC.value,
            content="Authentication spec",
        )
        spec2 = NodeData.create(
            type=NodeType.SPEC.value,
            content="Payment spec",
        )
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="Shared utility code",
        )

        self.db.add_node(spec1)
        self.db.add_node(spec2)
        self.db.add_node(code)

        # Link message to nodes
        self.db.link_message_to_node(message.id, spec1.id)
        self.db.link_message_to_node(message.id, spec2.id)
        self.db.link_message_to_node(message.id, code.id)

        # Get nodes from message
        nodes = self.db.get_nodes_from_message(message.id)

        assert len(nodes) == 3
        node_ids = {n.id for n in nodes}
        assert spec1.id in node_ids
        assert spec2.id in node_ids
        assert code.id in node_ids

    def test_get_nodes_from_nonexistent_message(self):
        """Test get_nodes_from_message with non-existent message."""
        nodes = self.db.get_nodes_from_message("nonexistent_msg")
        assert nodes == []

    def test_update_node_dialogue_metadata_basic(self):
        """Test updating dialogue metadata in node.data."""
        node = NodeData.create(
            type=NodeType.SPEC.value,
            content="User authentication specification",
        )
        self.db.add_node(node)

        # Update dialogue metadata
        self.db.update_node_dialogue_metadata(
            node.id,
            dialogue_turn_id="turn_5",
            definition_turn="turn_5",
            message_ids=["msg_1", "msg_2"],
        )

        # Verify updates
        updated_node = self.db.get_node(node.id)
        assert updated_node.data["dialogue_turn_id"] == "turn_5"
        assert updated_node.data["definition_turn"] == "turn_5"
        assert updated_node.data["message_ids"] == ["msg_1", "msg_2"]

    def test_update_node_dialogue_metadata_merge_references(self):
        """Test that referenced_in_turns merges correctly."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="def login(): pass",
        )
        self.db.add_node(node)

        # First update
        self.db.update_node_dialogue_metadata(
            node.id,
            referenced_in_turns=["turn_1", "turn_3"]
        )

        # Second update with overlapping turns
        self.db.update_node_dialogue_metadata(
            node.id,
            referenced_in_turns=["turn_3", "turn_5"]
        )

        # Verify merge and deduplication
        updated_node = self.db.get_node(node.id)
        turns = updated_node.data["referenced_in_turns"]
        assert len(turns) == 3
        assert set(turns) == {"turn_1", "turn_3", "turn_5"}

    def test_update_node_dialogue_metadata_hover_info(self):
        """Test updating hover metadata."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="Payment processing code",
        )
        self.db.add_node(node)

        hover_data = {
            "phase": "building",
            "created_by": "BUILDER",
            "status": NodeStatus.PENDING.value,
            "teleology_status": "justified",
            "related_nodes": ["spec_123", "test_456"],
            "key_findings": ["Needs error handling", "Consider retry logic"]
        }

        self.db.update_node_dialogue_metadata(
            node.id,
            hover_metadata=hover_data
        )

        updated_node = self.db.get_node(node.id)
        assert updated_node.data["hover_metadata"] == hover_data

    def test_update_node_dialogue_metadata_hover_merge(self):
        """Test that hover metadata merges correctly."""
        node = NodeData.create(
            type=NodeType.SPEC.value,
            content="API specification",
        )
        self.db.add_node(node)

        # First update
        self.db.update_node_dialogue_metadata(
            node.id,
            hover_metadata={"phase": "planning", "created_by": "ARCHITECT"}
        )

        # Second update with additional fields
        self.db.update_node_dialogue_metadata(
            node.id,
            hover_metadata={"status": "VERIFIED", "key_findings": ["Complete"]}
        )

        updated_node = self.db.get_node(node.id)
        hover = updated_node.data["hover_metadata"]
        assert hover["phase"] == "planning"
        assert hover["created_by"] == "ARCHITECT"
        assert hover["status"] == "VERIFIED"
        assert hover["key_findings"] == ["Complete"]

    def test_update_node_dialogue_metadata_nonexistent_node(self):
        """Test updating dialogue metadata for non-existent node raises error."""
        with pytest.raises(NodeNotFoundError):
            self.db.update_node_dialogue_metadata(
                "nonexistent_node",
                dialogue_turn_id="turn_1"
            )

    def test_get_node_hover_metadata_existing(self):
        """Test getting hover metadata that was previously set."""
        node = NodeData.create(
            type=NodeType.CODE.value,
            content="Login function",
            data={
                "hover_metadata": {
                    "phase": "testing",
                    "status": NodeStatus.TESTED.value,
                    "key_findings": ["All tests passing"]
                }
            }
        )
        self.db.add_node(node)

        hover = self.db.get_node_hover_metadata(node.id)
        assert hover["phase"] == "testing"
        assert hover["status"] == NodeStatus.TESTED.value
        assert hover["key_findings"] == ["All tests passing"]

    def test_get_node_hover_metadata_computed(self):
        """Test that hover metadata is computed if not present."""
        # Create a CODE node with some neighbors
        code = NodeData.create(
            type=NodeType.CODE.value,
            content="def process(): pass",
            status=NodeStatus.TESTING.value,
            created_by="BUILDER",
        )
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="Process specification",
        )
        test = NodeData.create(
            type=NodeType.TEST_SUITE.value,
            content="Test suite for process",
        )

        self.db.add_node(code)
        self.db.add_node(spec)
        self.db.add_node(test)

        # Add edges
        edge1 = EdgeData.implements(code.id, spec.id)
        edge2 = EdgeData.create(test.id, code.id, EdgeType.TESTS.value)
        self.db.add_edge(edge1)
        self.db.add_edge(edge2)

        # Get computed hover metadata
        hover = self.db.get_node_hover_metadata(code.id)

        assert hover["phase"] == "testing"  # Inferred from status
        assert hover["created_by"] == "BUILDER"
        assert hover["status"] == NodeStatus.TESTING.value
        assert len(hover["related_nodes"]) == 2
        assert spec.id in hover["related_nodes"]
        assert test.id in hover["related_nodes"]

    def test_get_node_hover_metadata_nonexistent_node(self):
        """Test getting hover metadata for non-existent node returns empty dict."""
        hover = self.db.get_node_hover_metadata("nonexistent_id")
        assert hover == {}

    def test_infer_phase_from_node(self):
        """Test phase inference for various node types."""
        # Test REQ node
        req = NodeData.create(type=NodeType.REQ.value, content="User requirement")
        self.db.add_node(req)
        phase = self.db._infer_phase_from_node(self.db.get_node(req.id))
        assert phase == "requirements"

        # Test RESEARCH node
        research = NodeData.create(type=NodeType.RESEARCH.value, content="Research findings")
        self.db.add_node(research)
        phase = self.db._infer_phase_from_node(self.db.get_node(research.id))
        assert phase == "research"

        # Test SPEC node
        spec = NodeData.create(type=NodeType.SPEC.value, content="Specification")
        self.db.add_node(spec)
        phase = self.db._infer_phase_from_node(self.db.get_node(spec.id))
        assert phase == "planning"

        # Test CODE node in TESTING status
        code_testing = NodeData.create(
            type=NodeType.CODE.value,
            content="Code",
            status=NodeStatus.TESTING.value
        )
        self.db.add_node(code_testing)
        phase = self.db._infer_phase_from_node(self.db.get_node(code_testing.id))
        assert phase == "testing"

        # Test CODE node in PENDING status
        code_building = NodeData.create(
            type=NodeType.CODE.value,
            content="Code",
            status=NodeStatus.PENDING.value
        )
        self.db.add_node(code_building)
        phase = self.db._infer_phase_from_node(self.db.get_node(code_building.id))
        assert phase == "building"

        # Test DOC node
        doc = NodeData.create(type=NodeType.DOC.value, content="Documentation")
        self.db.add_node(doc)
        phase = self.db._infer_phase_from_node(self.db.get_node(doc.id))
        assert phase == "documentation"

    def test_link_message_with_metadata(self):
        """Test linking message to node with custom metadata."""
        message = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Approve this specification",
        )
        spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="API specification",
        )

        self.db.add_node(message)
        self.db.add_node(spec)

        custom_metadata = {
            "action": "approval_request",
            "priority": "high",
            "reviewer": "human_user"
        }

        self.db.link_message_to_node(
            message.id,
            spec.id,
            metadata=custom_metadata
        )

        edge = self.db.get_edge(message.id, spec.id)
        assert edge.metadata["action"] == "approval_request"
        assert edge.metadata["priority"] == "high"
        assert edge.metadata["reviewer"] == "human_user"

    def test_message_node_bidirectional_lookup(self):
        """Test bidirectional lookup between messages and nodes."""
        # Create message and multiple nodes
        message = NodeData.create(
            type=NodeType.MESSAGE.value,
            content="Discussion about auth and payment",
        )
        auth_spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="Authentication spec",
        )
        payment_spec = NodeData.create(
            type=NodeType.SPEC.value,
            content="Payment spec",
        )

        self.db.add_node(message)
        self.db.add_node(auth_spec)
        self.db.add_node(payment_spec)

        # Link message to both specs
        self.db.link_message_to_node(message.id, auth_spec.id)
        self.db.link_message_to_node(message.id, payment_spec.id)

        # Test forward lookup (message -> nodes)
        nodes = self.db.get_nodes_from_message(message.id)
        assert len(nodes) == 2
        node_ids = {n.id for n in nodes}
        assert auth_spec.id in node_ids
        assert payment_spec.id in node_ids

        # Test reverse lookup (node -> messages)
        messages_for_auth = self.db.get_messages_for_node(auth_spec.id)
        assert len(messages_for_auth) == 1
        assert messages_for_auth[0].id == message.id

        messages_for_payment = self.db.get_messages_for_node(payment_spec.id)
        assert len(messages_for_payment) == 1
        assert messages_for_payment[0].id == message.id
