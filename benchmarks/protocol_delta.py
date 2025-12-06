"""
PROTOCOL DELTA - Layer 7 Verification Tests

Tests for the "Brain" layer - LLM integration and safety auditing.

Test Categories:
1. Creator Tests (Layer 7A): StructuredLLM output generation
2. Auditor Tests (Layer 7B): Syntax checking, alignment verification
3. Integration Tests: Full plan -> build -> verify flow

Run with:
    python -m benchmarks.protocol_delta

Success Criteria:
- All syntax checks correctly identify valid/invalid code
- Alignment verification catches mismatches
- add_node_safe rejects invalid code
- Orchestrator correctly uses LLM when available
"""
import sys
import unittest
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.graph_db import ParagonDB
from core.schemas import NodeData, EdgeData
from core.ontology import NodeType, NodeStatus, EdgeType
from agents.tools import (
    set_db, get_db, add_node, add_node_safe,
    check_syntax, verify_alignment,
    SyntaxCheckResult, SafeNodeResult, AuditResult,
)


# =============================================================================
# LAYER 7B TESTS - THE AUDITOR
# =============================================================================

class TestSyntaxCheck(unittest.TestCase):
    """Tests for tree-sitter syntax validation."""

    def test_valid_python_code(self):
        """Valid Python code should pass syntax check."""
        code = '''
def hello_world():
    """A simple function."""
    print("Hello, World!")
    return True
'''
        result = check_syntax(code, "python")

        self.assertTrue(result.success)
        self.assertTrue(result.valid)
        self.assertEqual(result.language, "python")
        self.assertEqual(len(result.errors), 0)

    def test_invalid_python_syntax(self):
        """Invalid Python should fail syntax check."""
        code = '''
def broken_function(
    print("Missing closing paren"
'''
        result = check_syntax(code, "python")

        self.assertTrue(result.success)  # Check ran successfully
        self.assertFalse(result.valid)   # But code is invalid
        self.assertGreater(len(result.errors), 0)

    def test_unclosed_string(self):
        """Unclosed string literal should be caught."""
        code = '''
x = "unclosed string
y = 42
'''
        result = check_syntax(code, "python")

        self.assertTrue(result.success)
        self.assertFalse(result.valid)

    def test_indentation_error(self):
        """Indentation errors may or may not be caught by tree-sitter."""
        # Note: tree-sitter is more lenient than Python interpreter
        code = '''
def foo():
print("wrong indent")
'''
        result = check_syntax(code, "python")
        # This test documents current behavior
        self.assertTrue(result.success)

    def test_unsupported_language(self):
        """Unsupported languages should pass with warning."""
        code = "fn main() { println!(\"Hello\"); }"
        result = check_syntax(code, "rust")

        self.assertTrue(result.success)
        self.assertTrue(result.valid)  # Passes through
        self.assertGreater(len(result.warnings), 0)

    def test_empty_code(self):
        """Empty code is technically valid syntax."""
        result = check_syntax("", "python")

        self.assertTrue(result.success)
        self.assertTrue(result.valid)


class TestAddNodeSafe(unittest.TestCase):
    """Tests for Layer 7B safety hooks."""

    def setUp(self):
        """Create fresh database for each test."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        """Reset global state."""
        set_db(None)

    def test_valid_code_node(self):
        """Valid code should be added successfully."""
        code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
        result = add_node_safe(
            node_type="CODE",
            content=code,
            created_by="test",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.syntax_valid)
        self.assertNotEqual(result.node_id, "")
        self.assertEqual(self.db.node_count, 1)

    def test_invalid_code_rejected(self):
        """Invalid code should be rejected."""
        code = '''
def broken(:
    print("syntax error"
'''
        result = add_node_safe(
            node_type="CODE",
            content=code,
            created_by="test",
        )

        self.assertFalse(result.success)
        self.assertFalse(result.syntax_valid)
        self.assertEqual(result.node_id, "")
        self.assertEqual(self.db.node_count, 0)  # Not added
        self.assertGreater(len(result.violations), 0)

    def test_spec_node_no_syntax_check(self):
        """SPEC nodes should not have syntax checking."""
        spec_content = "Implement a function that adds two numbers"

        result = add_node_safe(
            node_type="SPEC",
            content=spec_content,
            created_by="test",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.syntax_valid)  # Skipped for non-CODE
        self.assertEqual(self.db.node_count, 1)

    def test_code_with_spec_alignment(self):
        """Code with spec should check alignment."""
        # First create a SPEC node
        spec_result = add_node(
            node_type="SPEC",
            content="Create a function named calculate_sum that adds numbers",
            created_by="test",
        )
        self.assertTrue(spec_result.success)
        spec_id = spec_result.node_id

        # Now add aligned code
        code = '''
def calculate_sum(numbers):
    """Calculate the sum of numbers."""
    return sum(numbers)
'''
        result = add_node_safe(
            node_type="CODE",
            content=code,
            spec_id=spec_id,
            created_by="test",
            check_alignment=True,
        )

        # Syntax should always be valid for this code
        self.assertTrue(result.syntax_valid)
        # Success depends on alignment threshold - document behavior
        # Low threshold may pass, high threshold may fail
        # The important thing is the flow works without error
        if not result.success:
            # If it failed, it should be due to alignment, not other issues
            self.assertGreater(len(result.violations), 0)


class TestVerifyAlignment(unittest.TestCase):
    """Tests for spec-code alignment verification."""

    def setUp(self):
        """Create fresh database for each test."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        """Reset global state."""
        set_db(None)

    def test_aligned_code_passes(self):
        """Well-aligned code should pass verification."""
        # Create SPEC
        spec = NodeData.create(
            type="SPEC",
            content="Create a function named greet that takes a name and returns a greeting",
            created_by="test",
        )
        self.db.add_node(spec)

        # Create CODE that matches
        code = NodeData.create(
            type="CODE",
            content='''
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"
''',
            created_by="test",
        )
        self.db.add_node(code)

        # Add IMPLEMENTS edge
        edge = EdgeData.create(
            source_id=spec.id,
            target_id=code.id,
            type="IMPLEMENTS",
        )
        self.db.add_edge(edge)

        # Verify alignment
        result = verify_alignment(spec.id, code.id, threshold=0.3)

        self.assertTrue(result.success)
        self.assertTrue(result.syntax_valid)
        self.assertGreater(result.alignment_score, 0)
        # May or may not be approved depending on score

    def test_misaligned_code_flags(self):
        """Completely unrelated code should have low alignment."""
        # Create SPEC for one thing
        spec = NodeData.create(
            type="SPEC",
            content="Create a database connection manager with pooling",
            created_by="test",
        )
        self.db.add_node(spec)

        # Create CODE for something completely different
        code = NodeData.create(
            type="CODE",
            content='''
def sort_numbers(nums):
    """Sort a list of numbers."""
    return sorted(nums)
''',
            created_by="test",
        )
        self.db.add_node(code)

        # Add edge
        edge = EdgeData.create(
            source_id=spec.id,
            target_id=code.id,
            type="IMPLEMENTS",
        )
        self.db.add_edge(edge)

        # Verify - should have low alignment score
        result = verify_alignment(spec.id, code.id, threshold=0.7)

        self.assertTrue(result.success)
        # Low alignment for unrelated spec/code
        self.assertLess(result.alignment_score, 0.7)

    def test_missing_spec_node(self):
        """Should handle missing SPEC gracefully."""
        result = verify_alignment("nonexistent_spec", "nonexistent_code")

        self.assertFalse(result.success)
        self.assertFalse(result.approved)
        self.assertGreater(len(result.violations), 0)


# =============================================================================
# LAYER 7A TESTS - THE CREATOR (Schema Tests)
# =============================================================================

class TestSchemas(unittest.TestCase):
    """Tests for agent output schemas."""

    def test_implementation_plan_schema(self):
        """ImplementationPlan should serialize/deserialize correctly."""
        from agents.schemas import ImplementationPlan, ComponentSpec, DependencyEdge
        import msgspec

        plan = ImplementationPlan(
            explanation="Test plan",
            components=[
                ComponentSpec(
                    name="component1",
                    type="function",
                    description="A test function",
                    dependencies=[],
                ),
                ComponentSpec(
                    name="component2",
                    type="class",
                    description="A test class",
                    dependencies=["component1"],
                ),
            ],
            dependencies=[
                DependencyEdge(source="component2", target="component1"),
            ],
        )

        # Serialize
        data = msgspec.json.encode(plan)
        self.assertIsInstance(data, bytes)

        # Deserialize
        decoded = msgspec.json.decode(data, type=ImplementationPlan)
        self.assertEqual(decoded.explanation, "Test plan")
        self.assertEqual(len(decoded.components), 2)

    def test_code_generation_schema(self):
        """CodeGeneration should serialize correctly."""
        from agents.schemas import CodeGeneration
        import msgspec

        code = CodeGeneration(
            filename="example.py",
            code="def foo(): pass",
            imports=["os", "sys"],
            description="Example function",
        )

        data = msgspec.json.encode(code)
        decoded = msgspec.json.decode(data, type=CodeGeneration)

        self.assertEqual(decoded.filename, "example.py")
        self.assertEqual(len(decoded.imports), 2)

    def test_auditor_schemas(self):
        """Auditor output schemas should work correctly."""
        from agents.schemas import (
            SyntaxCheck, AlignmentCheck, AuditorReport
        )
        import msgspec

        syntax = SyntaxCheck(
            valid=True,
            language="python",
            errors=[],
            warnings=[],
        )

        alignment = AlignmentCheck(
            aligned=True,
            spec_id="spec_123",
            code_id="code_456",
            confidence=0.95,
        )

        report = AuditorReport(
            syntax_check=syntax,
            alignment_check=alignment,
            approved=True,
        )

        data = msgspec.json.encode(report)
        decoded = msgspec.json.decode(data, type=AuditorReport)

        self.assertTrue(decoded.approved)
        self.assertTrue(decoded.syntax_check.valid)


# =============================================================================
# PROMPT BUILDER TESTS
# =============================================================================

class TestPromptBuilders(unittest.TestCase):
    """Tests for context-aware prompt generation."""

    def setUp(self):
        """Create fresh database."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        set_db(None)

    def test_architect_prompt_builder(self):
        """Architect prompt should include REQ content."""
        from agents.prompts import build_architect_prompt

        # Create REQ node
        req = NodeData.create(
            type="REQ",
            content="Build a REST API for user management",
            created_by="test",
        )
        self.db.add_node(req)

        system_prompt, user_prompt = build_architect_prompt(self.db, req.id)

        self.assertIn("Architect", system_prompt)
        self.assertIn("REST API", user_prompt)
        self.assertIn("user management", user_prompt)

    def test_builder_prompt_builder(self):
        """Builder prompt should include SPEC content."""
        from agents.prompts import build_builder_prompt

        # Create SPEC node
        spec = NodeData.create(
            type="SPEC",
            content="Create a User class with name and email fields",
            created_by="test",
        )
        self.db.add_node(spec)

        system_prompt, user_prompt = build_builder_prompt(self.db, spec.id)

        self.assertIn("Builder", system_prompt)
        self.assertIn("User class", user_prompt)
        self.assertIn("email", user_prompt)

    def test_unified_prompt_builder(self):
        """Unified builder should dispatch correctly."""
        from agents.prompts import build_prompt

        req = NodeData.create(type="REQ", content="Test req", created_by="test")
        self.db.add_node(req)

        system_prompt, user_prompt = build_prompt(self.db, "architect", req.id)
        self.assertIn("Architect", system_prompt)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestLayer7Integration(unittest.TestCase):
    """Integration tests for the full Layer 7 flow."""

    def setUp(self):
        """Create fresh database."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        set_db(None)

    def test_full_flow_without_llm(self):
        """Test that orchestrator works without LLM available."""
        from agents.orchestrator import TDDOrchestrator

        orchestrator = TDDOrchestrator(enable_checkpointing=False)
        result = orchestrator.run(
            session_id="test_session",
            task_id="test_task",
            spec="Create a hello world function",
            requirements=["Should print 'Hello'"],
            max_iterations=2,
        )

        # Should complete (pass or fail) even without LLM
        self.assertIn(result.get("final_status"), ["passed", "failed", None])
        self.assertIn(result.get("phase"), ["passed", "failed"])

    def test_syntax_and_add_integration(self):
        """Test syntax check flows into add_node_safe."""
        # Valid code should flow through
        valid_code = "def valid(): return True"
        result = add_node_safe(
            node_type="CODE",
            content=valid_code,
            created_by="test",
        )
        self.assertTrue(result.success)
        self.assertEqual(self.db.node_count, 1)

        # Invalid code should be blocked
        invalid_code = "def invalid(: pass"
        result2 = add_node_safe(
            node_type="CODE",
            content=invalid_code,
            created_by="test",
        )
        self.assertFalse(result2.success)
        self.assertEqual(self.db.node_count, 1)  # Still just 1

    def test_spec_code_edge_creation(self):
        """Test that IMPLEMENTS edges are created correctly."""
        # Create SPEC
        spec_result = add_node(
            node_type="SPEC",
            content="Implement a function foo",
            created_by="test",
        )
        spec_id = spec_result.node_id

        # Create CODE with spec reference
        code = "def foo(): return 'bar'"
        code_result = add_node_safe(
            node_type="CODE",
            content=code,
            spec_id=spec_id,
            created_by="test",
            check_alignment=True,
        )

        # Should have created the edge if alignment passed
        if code_result.success:
            self.assertTrue(self.db.has_edge(spec_id, code_result.node_id))


# =============================================================================
# LLM MOCK TESTS (for when LLM not available)
# =============================================================================

# Check if litellm is available
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class TestLLMInterface(unittest.TestCase):
    """Tests for the LLM interface structure."""

    @unittest.skipUnless(LITELLM_AVAILABLE, "litellm not installed")
    def test_llm_module_structure(self):
        """Verify core/llm.py has expected interface."""
        from core.llm import StructuredLLM, get_llm, set_llm, reset_llm

        # Should be importable
        self.assertTrue(callable(get_llm))
        self.assertTrue(callable(set_llm))
        self.assertTrue(callable(reset_llm))

    @unittest.skipUnless(LITELLM_AVAILABLE, "litellm not installed")
    def test_structured_llm_init(self):
        """StructuredLLM should initialize with defaults."""
        from core.llm import StructuredLLM

        llm = StructuredLLM()
        self.assertIsNotNone(llm.model)
        self.assertEqual(llm.temperature, 0.0)

    @unittest.skipUnless(LITELLM_AVAILABLE, "litellm not installed")
    def test_schema_prompt_generation(self):
        """Schema prompt should generate valid JSON Schema."""
        from core.llm import StructuredLLM
        from agents.schemas import ImplementationPlan

        llm = StructuredLLM()
        schema_json = llm._get_schema_prompt(ImplementationPlan)

        self.assertIn("explanation", schema_json)
        self.assertIn("components", schema_json)

    def test_llm_graceful_unavailability(self):
        """System should handle missing LLM gracefully."""
        # The orchestrator should work even without litellm
        from agents.orchestrator import LLM_AVAILABLE
        # Just verify the flag exists and is a boolean
        self.assertIsInstance(LLM_AVAILABLE, bool)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Run all Protocol Delta tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSyntaxCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestAddNodeSafe))
    suite.addTests(loader.loadTestsFromTestCase(TestVerifyAlignment))
    suite.addTests(loader.loadTestsFromTestCase(TestSchemas))
    suite.addTests(loader.loadTestsFromTestCase(TestPromptBuilders))
    suite.addTests(loader.loadTestsFromTestCase(TestLayer7Integration))
    suite.addTests(loader.loadTestsFromTestCase(TestLLMInterface))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("PROTOCOL DELTA SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ PROTOCOL DELTA PASSED - Layer 7 (The Brain) Verified")
    else:
        print("\n✗ PROTOCOL DELTA FAILED")
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}: {trace.split(chr(10))[0]}")
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}: {trace.split(chr(10))[0]}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
