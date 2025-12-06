"""
PARAGON SECURITY TESTS

Security test suite for code injection and prompt injection resistance.

Test Categories:
1. Code Injection - Testing tree-sitter validation against dangerous patterns
2. Prompt Injection - Testing prompt sanitization and structure preservation

Run with:
    python -m pytest tests/security/
    python -m pytest tests/security/test_code_injection.py
    python -m pytest tests/security/test_prompt_injection.py
"""
